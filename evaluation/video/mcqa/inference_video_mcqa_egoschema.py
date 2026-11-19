import os
import re
import math
import json
import argparse
import warnings
import traceback

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


class EgoschemaDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list, processor):
        self.data_folder = data_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.data_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_tensor = self.processor(video_path)

        return {
            'q_uid': q_uid,
            'video': video_tensor, 
            'record': line,
        }


def collate_fn(batch):
    v_id = [x['q_uid'] for x in batch]
    vid = [x['video'] for x in batch]
    rcs = [x['record'] for x in batch]
    return v_id, vid, rcs


def egoschema_dump(q_uid, instruct, options, output):
    letters = ['A', 'B', 'C', 'D', 'E']

    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(q_uid, instruct, output)
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

    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = EgoschemaDataset(args.video_folder, questions, processor['video'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Iterate over each sample in the ground truth file
    for i, (vids, videos, records) in enumerate(tqdm(dataloader)):
        q_uid = vids[0]
        video_tensor = videos[0]
        line = records[0]

        question = line['question']
        a0 = line['option 0']
        a1 = line['option 1']
        a2 = line['option 2']
        a3 = line['option 3']
        a4 = line['option 4']
        options = [a0, a1, a2, a3, a4]

        instruct = f'Select the best answer to the following multiple-choice question based on the video.\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: ' 

        try:
            pred = mm_infer(
                video_tensor,
                instruct,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            traceback.print_exc()
            pred = 'C'

        pred_idx = egoschema_dump(q_uid, instruct, options, pred)
        ans_file.write(f'{q_uid}, {pred_idx}\n')

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
