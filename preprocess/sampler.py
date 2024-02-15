import os
import argparse
import random
import glob
from preprocess.loader import FactoryDataLoader
from sleep_audio.core import write_audio, respiration_estimate_from_peakdetect

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="/data1/ryan/separation/audio_raw/")
    parser.add_argument("--output_path", type=str, default="", required=True)
    args = parser.parse_args()
    return args


def sampling(raw_data, split_path):
    print("Start load id: ", raw_data, ", split: ", split_path)
    loader = FactoryDataLoader().loader(int(raw_data))
    data, sr = loader.load(os.path.join(args.raw_path, raw_data + '_data'))
    data = data[5:-2]
    print("Complete load id: ", raw_data)
    j = 0
    for c_audio in data:
        for i in range(10):
            duration = sr * 30
            source = c_audio[i*duration:(i+1)*duration]
            if len(source) < duration:
                continue

            PDV = respiration_estimate_from_peakdetect(source)
            if PDV < 0.4:
                continue

            # save clean and noise audio
            output_source_path = os.path.join(args.output_path, 'source', 
                                              split_path, 'source_' + raw_data + '_' + str(j) + '.wav')
            write_audio(output_source_path, source, 16000)
            j = j + 1


if __name__ == "__main__":
    args = get_args()
    
    '''
        sampling from candidate raw data
        1. load data -> return audio data splited 5 minutes 
        2. when peak detection value is over 0.4,
        3. save source data each 30 seconds
    '''

    data_path = ['train', 'test']
    if os.path.isdir(args.output_path) is not True:
        os.mkdir(args.output_path)
        os.mkdir(os.path.join(args.output_path, 'source'))
        for path in data_path:
            os.mkdir(os.path.join(args.output_path, 'source', path))

    file_list = glob.glob(args.raw_path + '/50*')
    candidate_raw_data = []
    for path in file_list:
        if not os.path.isdir(path):
            continue
        session_id = path.split('/')[-1].split('_')[0]
        if int(session_id) > 0:
            candidate_raw_data.append(session_id)

    file_list = glob.glob(args.raw_path + '/0*')
    for path in file_list:
        if not os.path.isdir(path):
            continue
        session_id = path.split('/')[-1].split('_')[0]
        if int(session_id) > 0:
            candidate_raw_data.append(session_id)

    random.shuffle(candidate_raw_data)
    print(candidate_raw_data)

    train_data = candidate_raw_data[:-12]
    test_data = candidate_raw_data[-12:]

    dataset = [train_data, test_data]

    for data, d_name in zip(dataset, data_path):
        for d in data:
            sampling(d, d_name)
