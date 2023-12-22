import csv
import argparse
import os
import torch
import torch.nn.functional as F

from sleep_audio import load, write_audio
from separation.sf_separator import separation_nn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="./eval.tsv")
    parser.add_argument("--model_name", type=str, default="cmgan")
    parser.add_argument("--test_name", type=str, default="eval")
    args = parser.parse_args()
    return args

def pad_zero(tensor):
    target_size = 480000
    current_size = tensor.size(0)
    padding_size = target_size - current_size

    if padding_size > 0:
        tensor = F.pad(tensor, pad=(0, padding_size), mode='constant', value=0)

    assert target_size == tensor.size(0)

    return tensor

if __name__ == "__main__":
    args = get_args()
    
    # TSV file path
    file_path = args.test_path
    sr = 16000

    # sep = Separator()
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            # id	Near type	Far type	Near session id	Far session id	file_name	start time(sec)	end time(sec)   Expect Result
            # T00	None	None	143	144	00_00_00_00	240	270	None	None
            print(row)
            test_id = row[0]
            session_id = row[3]
            clionic_file_name = row[5] + '.part.mp3'
            start_time = int(row[6])
            end_time = start_time + 30
            clionic_file_path = os.path.join('/nas/raw-data/clionic_hospital_psg/', 'CLN0' + session_id, 'micdata', clionic_file_name)
            
            audio = load(clionic_file_path)
            raw_audio = audio[start_time*sr:end_time*sr]
            
            eval_save_path = os.path.join('./', args.test_name)
            if not os.path.exists(eval_save_path):
                os.mkdir(eval_save_path)

            raw_audio = torch.Tensor(raw_audio)
            raw_audio = pad_zero(raw_audio)
            est1, est2 = separation_nn(raw_audio, test_id)
