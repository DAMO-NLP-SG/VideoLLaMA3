import os
import re
import math
import json
import queue
import random
import argparse
import warnings
import threading
import traceback

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from evaluation.register import INFERENCES
from videollama3.utils import disable_torch_init

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class CUDADataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.stream = torch.cuda.Stream(local_rank) # create a new cuda stream in each process
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            frames = self.batch['video'][0][0][0]
            for idx, frame in enumerate(frames):
                frames[idx]['pixel_values'] = frame['pixel_values'].to(device=self.local_rank, non_blocking=True)
                frames[idx]['image_grid_thw'] = frame['image_grid_thw'].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()
        

class MLVUDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def qa_template(self, question, candidates, answer):
        question = f"Question: {question}\n"
        question += "Options:\n"
        answer_idx = -1
        for idx, c in enumerate(candidates):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip() + '\n'
        question += "Answer with the option\'s letter from the given choices directly and only give the best option."
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer

    def __getitem__(self, idx):
        task_type = self.data_list[idx]['task_type']
        
        video_folder = self.data_list[idx]['prefix']
        video_name = self.data_list[idx]['data']['video']
        video_path = os.path.join(video_folder, video_name)
        
        question = self.data_list[idx]['data']['question']
        candidates = self.data_list[idx]['data']['candidates']
        answer = self.data_list[idx]['data']['answer']
        instruct, answer = self.qa_template(question, candidates, answer)

        video_tensor = self.processor(video_path)

        return {
            'video_name': video_name,
            'video':      video_tensor,
            'instruct':   instruct, 
            'candidates': candidates,
            'answer':     answer,
            'task_type':  task_type,
        }


def collate_fn(batch):
    return {
        'video_name': [x['video_name'] for x in batch], 
        'video':      [x['video'] for x in batch],
        'instruct':   [x['instruct'] for x in batch],
        'candidates': [x['candidates'] for x in batch],
        'answer':     [x['answer'] for x in batch],
        'task_type':  [x['task_type'] for x in batch],
    }


mcqa_tasks = {
    "plotQA": ("1_plotQA.json", "1_plotQA"),
    "needle": ("2_needle.json", "2_needle"),
    "ego":    ("3_ego.json",    "3_ego"),
    "count":  ("4_count.json",  "4_count"),
    "order":  ("5_order.json",  "5_order"),
    "anomaly_reco": ("6_anomaly_reco.json", "6_anomaly_reco"),
    "topic_reasoning": ("7_topic_reasoning.json", "7_topic_reasoning")
}

oqa_tasks = {
    "subPlot": ("8_sub_scene.json", "8_sub_scene"),
    "summary": ("9_summary.json", "9_summary")
}


def build_mlvu_eval(args, processor):
    data_list = []
    for task_name, task in mcqa_tasks.items():
        json_file = os.path.join(args.question_file, task[0])
        vis_folder = os.path.join(args.video_folder, task[1])
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            data_list.append({
                'task_type': task_name,
                'prefix': vis_folder,
                'data': data
            })
    # set random seed
    random.seed(42)
    random.shuffle(data_list)
    data_list  = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    dataset    = MLVUDataset(data_list, processor)
    dataloader = CUDADataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    if pred==gt_option:
        flag=True

    return flag


def mlvu_dump(candidates, output):
    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    
    letters = [chr(i) for i in range(ord('A'), ord('A') + len(candidates))]
    
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    if len(pred_answer) == 0:
        pred_answer = 'C'
    else:
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip('()')
    
    return pred_answer


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor, tokenizer = model_init(args.model_path)

    answer_file = args.answer_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')

    output_dir = os.path.dirname(answer_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_mlvu_eval(args, processor['video'])

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        video_name   = line['video_name'][0]
        video_tensor = line['video'][0]
        task_type    = line['task_type'][0]
        instruct     = line['instruct'][0]
        candidates   = line['candidates'][0]
        answer       = line['answer'][0]#.item()

        output = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
        )

        pred_answer = mlvu_dump(candidates, output)

        ans_file.write(json.dumps({"task_type": task_type, "video": video_name,  "pred": pred_answer, "gt": answer}) + '\n')

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
