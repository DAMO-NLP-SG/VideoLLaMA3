import os
import re
import math
import json
import queue
import random
import argparse
import warnings
import traceback
import threading

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from evaluation.register import INFERENCES
from videollama3 import disable_torch_init

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


class MVBenchDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bound = (None, None)
        if self.data_list[idx]['bound']:
            bound = (self.data_list[idx]['data']['start'], self.data_list[idx]['data']['end'])
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        video_tensor = self.processor(video_path, s=bound[0], e=bound[1])
        question     = self.data_list[idx]['data']['question']
        options      = self.data_list[idx]['data']['candidates']
        answer       = self.data_list[idx]['data']['answer']
        task_type    = self.data_list[idx]['task_type']

        answer_idx = -1
        letters = []
        options_string = ''
        for option_idx, c in enumerate(options):
            letters.append(f"{chr(ord('A') + option_idx)}")
            options_string += f"({chr(ord('A') + option_idx)}) {c}\n"
            if c == answer:
                answer_idx = option_idx

        instruct = f'Question: {question}\nOptions:\n{options_string}Answer with the option\'s letter from the given choices directly and only give the best option.' 

        return {
            'video':      video_path, 
            'video_data': video_tensor,
            'instruct':   instruct,
            'options':    options,
            'answer_idx': answer_idx,
            'task_type':  task_type
        }


def collate_fn(batch):
    return {
        'video':      [x['video'] for x in batch],
        'video_data': [x['video_data'] for x in batch],
        'instruct':   [x['instruct'] for x in batch],
        'options':    [x['options'] for x in batch],
        'answer_idx': [x['answer_idx'] for x in batch],
        'task_type':  [x['task_type'] for x in batch]
    }

tasks = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}


def build_mvbench_eval(args, processor):
    data_list = []
    for task_name, task in tasks.items():
        json_file = os.path.join(args.question_file, task[0])
        vis_folder = os.path.join(args.video_folder, task[1])
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            data_list.append({
                'task_type': task_name,
                'prefix': vis_folder,
                'data_type': task[2],
                'bound': task[3],
                'data': data
            })
    # set random seed
    random.seed(42)
    random.shuffle(data_list)
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    dataset = MVBenchDataset(data_list, processor)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    dataloader = CUDADataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


def mvbench_dump(instruct, options, output):

    letters = [chr(ord('A') + i) for i in range(len(options))]
    
    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(vid, instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2
    
    return pred_idx


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor, tokenizer = model_init(args.model_path)

    answer_file = args.answer_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')

    output_dir = os.path.dirname(answer_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_mvbench_eval(args, processor['video'])

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        video        = line['video'][0]
        video_tensor = line['video_data'][0]
        instruct     = line['instruct'][0]
        options      = line['options'][0]
        answer_idx   = line['answer_idx'][0]#.item()
        task_type    = line['task_type'][0]

        output = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
        )

        pred_idx = mvbench_dump(instruct, options, output)

        ans_file.write(json.dumps({"video": video, "task_type": task_type, "pred": pred_idx, "gt": answer_idx}) + '\n')

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
