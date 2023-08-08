import os
import argparse
import multiprocessing

from preprocess.loader import FactoryDataLoader
from preprocess.audio_preprocess import save_waveform, get_IRR_SNR, drc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    args = parser.parse_args()
    return args


def sampling(raw_data):
    print(raw_data)

    loader = FactoryDataLoader().loader(int(raw_data))
    data, sr = loader.load(os.path.join(args.raw_path, raw_data + '_data'))

    # mix every 30 sec
    j = 0
    for c_audio in data:
        source = drc(c_audio)
        for i in range(10):
            duration = sr * 30                    
            source = c_audio[i*duration:(i+1)*duration]
            source = loader.nr(source, sr)
            IRR = get_IRR_SNR(source, sr)
            if IRR < IRR_THRESHOLD:
                continue

            # save clean and noise audio
            output_source_path = os.path.join(args.output_path, 'source', 'source_' + raw_data + '_' + str(j) + '.wav')
            save_waveform(output_source_path, source, 16000)
            j = j + 1


if __name__ == "__main__":
    args = get_args()
    
    '''
        sampling from candidate raw data
        1. load data -> return audio data splited 5 minutes 
        2. preprocess: normalization and noise reduction
        3. save source data each 30 seconds
    '''

    IRR_THRESHOLD = 17.5

    if os.path.isdir(args.output_path) is not True:
        os.mkdir(args.output_path)
        os.mkdir(os.path.join(args.output_path, 'source'))

    candidate_raw_data = ['10002', '10003', '10004', '10005', '10006', '10007', '10009', '10010', '10011', '10012', '10013']
    RAW_DATA_PATH = args.raw_path

    pool = multiprocessing.Pool(processes=5)
    pool.map(sampling, candidate_raw_data)
    pool.close()
    pool.join()

    # save sample for check
