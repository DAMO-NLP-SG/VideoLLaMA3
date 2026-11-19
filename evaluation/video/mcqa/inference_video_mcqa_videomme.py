import os
import re
import math
import json
import copy
import queue
import random
import argparse
import warnings
import traceback
import threading

import cv2
import torch
import pysubs2
import numpy as np
import pyarrow.parquet as pq
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


# def get_seq_frames(total_num_frames, desired_num_frames):
#     """
#     Calculate the indices of frames to extract from a video.

#     Parameters:
#     total_num_frames (int): Total number of frames in the video.
#     desired_num_frames (int): Desired number of frames to extract.

#     Returns:
#     list: List of indices of frames to extract.
#     """

#     # Calculate the size of each segment from which a frame will be extracted
#     seg_size = float(total_num_frames - 1) / desired_num_frames

#     seq = []
#     for i in range(desired_num_frames):
#         # Calculate the start and end indices of each segment
#         start = int(np.round(seg_size * i))
#         end = int(np.round(seg_size * (i + 1)))

#         # Append the middle index of the segment to the list
#         seq.append((start + end) // 2)

#     return seq


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


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = seg_size * i
        end   = seg_size * (i + 1)

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return np.round(np.array(seq) + 1e-6).astype(int)


class VideoMMEDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, subtitle_folder, data_list, processor, num_frames=8):
        self.video_folder = video_folder
        self.subtitle_folder = subtitle_folder
        self.data_list = data_list
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]

        video_ytid = line['url'].split('watch?v=')[-1]

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.video_folder, f'{video_ytid}{fmt}')
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        subtitle_path = os.path.join(self.subtitle_folder, f'{video_ytid}.srt')

        try:
            video_tensor = self.processor(video_path)
            num_frames = self.num_frames
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_ytid}')
            video_tensor = None
            num_frames = 0

        if video_tensor is not None and os.path.exists(subtitle_path):
            frame_timestamps = video_tensor[1]

            subs = pysubs2.load(subtitle_path, encoding="utf-8")

            subtitles = []
            frame_idx = 0
            for sub in subs:
                if frame_idx == len(frame_timestamps):
                    break
                while frame_timestamps[frame_idx] < sub.start or frame_timestamps[frame_idx] > sub.end:
                    frame_idx += 1
                    if frame_idx == len(frame_timestamps):
                        break
                subtitles.append(sub.text.replace("\\N", " ").rstrip())

            subtitles = "\n".join(subtitles)
        else:
            subtitles = ""

        return {
            'video': video_tensor,
            'subtitle': subtitles,
            'record': line,
        }


def collate_fn(batch):
    return {
        'video':    [x['video'] for x in batch],
        'subtitle': [x['subtitle'] for x in batch],
        'record':   [x['record'] for x in batch],
    }


def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    # Convert PyArrow Table to pandas DataFrame
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append({
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "question": record.question,
                        "choices": list(record.options),
                        "answer": record.answer,
                    }
                ]
            })
        else:
            jsons[-1]['questions'].append({
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            })

    return jsons


def videomme_dump(record, instruct, options, output):
    letters = ['A', 'B', 'C', 'D']

    digit2word = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
        '0': 'zero',
    }

    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall('[\(\ \[]*([A-D])[\)\.\ \]]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                opt2 = opt
                if opt in digit2word:
                    opt2 = digit2word[opt]
                if opt.lower() in output.lower() or opt2.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(record['youtube_id'], instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2

    return letters[pred_idx]


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    answer_file = args.answer_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')
    answer_sub_file = answer_file.replace('.json', '_sub.json')

    output_dir = os.path.dirname(answer_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ans_file = open(answer_file, "w")
    ans_sub_file = open(answer_sub_file, "w")

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else 8

    # convert parquet to json
    questions = load_parquet(args.question_file)
    # set random seed
    random.seed(42)
    random.shuffle(questions)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoMMEDataset(args.video_folder, args.subtitle_folder, questions, processor['video'], num_frames)
    dataloader = CUDADataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # Iterate over each sample in the ground truth file
    for i, (videos, subtitles, records) in enumerate(tqdm(dataloader)):
        video_tensor = videos[0]
        subtitle = subtitles[0]
        record = records[0]

        new_record = copy.deepcopy(record)
        new_record_sub = copy.deepcopy(record)

        if video_tensor is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            new_record_sub['missing'] = True
            ans_sub_file.write(json.dumps(new_record_sub) + ",\n")
            continue
        else:
            new_record['missing'] = False
            new_record_sub['missing'] = False

        questions = record['questions']
        for idx, question in enumerate(questions):
            q = question['question']
            choices = question['choices']
            options = [re.findall('[A-D]\. (.*).', c)[0] for c in choices]

            instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            instruct += f"{q}\n"
            for cho_idx, cho in enumerate(choices):
                instruct += f"{cho}\n"
            # instruct += "The best option is: "
            instruct += "Answer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video', do_sample=False)
            new_record['questions'][idx]['response'] = videomme_dump(record, instruct, options, output)

            instruct_sub = f"This video's subtitles are listed below:\n{subtitle}\n" + instruct
            output_sub = mm_infer(video_tensor, instruct_sub, model=model, tokenizer=tokenizer, modal='video', do_sample=False)
            new_record_sub['questions'][idx]['response'] = videomme_dump(record, instruct_sub, options, output_sub)

        ans_file.write(json.dumps(new_record) + ",\n")
        ans_sub_file.write(json.dumps(new_record_sub) + ",\n")

    ans_file.close()
    ans_sub_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--subtitle-folder', help='Directory containing subtitle files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
