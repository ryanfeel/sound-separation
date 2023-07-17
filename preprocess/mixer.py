import argparse
import numpy as np
import soundfile as sf
import os
from itertools import permutations
import multiprocessing

from util.meta import create_custom_dataset
from preprocess.loader import FactoryDataLoader
from preprocess.snr_mixer import SNRMixer
from preprocess.room_simulation import RoomSimulator
from preprocess.audio_preprocess import save_waveform, normalize, noise_reduction, \
    noise_reduction_origin, get_IRR_SNR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dataset_name", type=str, default="v0.5")
    args = parser.parse_args()
    return args

def make_mix_eval_data(raw_data):
    mixer = SNRMixer()
    print(raw_data[0], raw_data[1])
    raw_data_1 = raw_data[0]
    raw_data_2 = raw_data[1]

    if int(raw_data_1) < 100:
        data_type = 'HomePSG'
    else:
        data_type = 'PSG'
    path_name = os.path.join(
        args.output_path,
        'result_' + data_type + '_' + raw_data_1 + '_' + raw_data_2 + '_snr_0'
    )
    print(path_name)
    if os.path.isdir(path_name):
        return

    clean_data, sr = FactoryDataLoader().loader(int(raw_data_1)).load(os.path.join(RAW_DATA_PATH, raw_data_1 + '_data'))
    noise_data, sr = FactoryDataLoader().loader(int(raw_data_2)).load(os.path.join(RAW_DATA_PATH, raw_data_2 + '_data'))

    # length check -> extract method
    if len(clean_data) < len(noise_data):
        noise_data = noise_data[:len(clean_data)]
    else:
        diff = len(clean_data)-len(noise_data)
        for i in range(diff):
            noise_data.append(noise_data[i])

    # mix every 30 sec
    k = 0
    print(len(clean_data), len(noise_data))
    for c_audio, n_audio in zip(clean_data, noise_data):
        c_audio = normalize(c_audio, sr)
        n_audio = normalize(n_audio, sr)
        mix_snr_list = [[], [], [], []]
        for i in range(10):
            duration = sr * 30
            if len(c_audio) < i*duration:
                continue
                
            source1 = c_audio[i*duration:(i+1)*duration]
            source2 = n_audio[i*duration:(i+1)*duration]
            
            if len(source1) < len(source2):
                source2 = source2[:len(source1)]
            else:
                diff = len(source1) - len(source2)
                # add pad 0 
                pad = []
                for i in range(diff):
                    pad.append(0.0)
                source2 = np.concatenate([source2, pad])
            
            source1 = noise_reduction(source1, sr)
            source2 = noise_reduction(source2, sr)

            for j in range(4):
                mix_snr_list[j] = np.concatenate([mix_snr_list[j], mixer.mix(source1, source2, j*6, sr)])

        # save mix
        i = 0
        for i in range(4):
            if int(raw_data_1) < 100:
                data_type = 'HomePSG'
            else:
                data_type = 'PSG'
            path_name = os.path.join(
                args.output_path,
                'result_' + data_type + '_' + raw_data_1 + '_' + raw_data_2 + '_snr_' + str(i*6)
            )
            if not os.path.isdir(path_name):
                os.mkdir(path_name)
            save_waveform(os.path.join(path_name, 'audio_' + str(k) + '.wav'), mix_snr_list[i], 16000)
        k = k + 1

def make_mix_data(raw_data):
    print(raw_data[0], raw_data[1])
    raw_data_1 = raw_data[0]
    raw_data_2 = raw_data[1]

    clean_data, sr = FactoryDataLoader().loader(int(raw_data_1)).load(os.path.join(RAW_DATA_PATH, raw_data_1 + '_data'))
    noise_data, sr = FactoryDataLoader().loader(int(raw_data_2)).load(os.path.join(RAW_DATA_PATH, raw_data_2 + '_data'))

    # length check -> extract method
    if len(clean_data) <= len(noise_data):
        noise_data = noise_data[:len(clean_data)]
    else:
        noise_data.append(noise_data[:(len(clean_data)-len(noise_data))])

    # mix every 30 sec
    # selecting 2 source (not same session id)
    j = 2
    for c_audio, n_audio in zip(clean_data[2:-2], noise_data[2:-2]):
        c_audio = normalize(c_audio, sr)
        n_audio = normalize(n_audio, sr)
        for i in range(10):
            duration = sr * 30                    
            source1 = c_audio[i*duration:(i+1)*duration]
            source2 = n_audio[i*duration:(i+1)*duration]
            source1 = noise_reduction(source1, sr)
            source2 = noise_reduction(source2, sr)

            if get_IRR_SNR(source1, sr) < 15 or get_IRR_SNR(source2, sr) < 15:
                continue

            c_file_name = '_' + str(j) + '_' + str(i)
            mixer = RoomSimulator()
            # get and save mixture
            output_mix_path = os.path.join(
                args.output_path, 
                'mixture', 
                'mix_'+ raw_data_1 + '_' + raw_data_2 +\
                    c_file_name + '.wav') 
            mixer.simulate(output_mix_path, source1, source2)

            mixer = RoomSimulator()
            output_mix_path2 = os.path.join(
                args.output_path, 
                'mixture', 
                'mix_'+ raw_data_2 + '_' + raw_data_1 +\
                    c_file_name + '.wav') 
            mixer.simulate(output_mix_path2, source2, source1)

            # save clean and noise audio
            output_source1_path = os.path.join(args.output_path, 'source1', 'source1_' + raw_data_1 + c_file_name + '.wav')
            output_source2_path = os.path.join(args.output_path, 'source2', 'source2_' + raw_data_2 + c_file_name + '.wav')
            save_waveform(output_source1_path, source1, 16000)
            save_waveform(output_source2_path, source2 , 16000)
        j = j + 1

if __name__ == "__main__":
    args = get_args()
    
    '''
        make mixing code
        1. load data -> return audio data splited 5 minutes 
        2. preprocess: normalization and noise reduction
        3. make 2 mixture as two snr ratio, randomize snr (6~18, 0)
        4. save mix, s0, s1 and meta data
    '''


    if os.path.isdir(args.output_path) is not True:
        os.mkdir(args.output_path)
        print(os.path.join(args.output_path, 'mixture'))
        os.mkdir(os.path.join(args.output_path, 'mixture'))
        os.mkdir(os.path.join(args.output_path, 'source1'))
        os.mkdir(os.path.join(args.output_path, 'source2'))

    # select raw data, TODO: will be load from db
    if args.mode == 'train':
        candidate_raw_data = ['033', '123', '010', '016', '017', '029',  '1324', '1459', '1495']
        RAW_DATA_PATH = args.raw_path

        pool = multiprocessing.Pool(processes=1)
        # TODO: 1) make pre-data read single wav -> find irr -> save wav 
        # 2) mix using pre-data
        pool.map(make_mix_data, permutations(candidate_raw_data, 2))
        pool.close()
        pool.join()

        # write meta data
        create_custom_dataset(
            args.output_path,
            args.output_path,
            args.dataset_name
        )
        # save sample for check

    elif args.mode == 'eval':
        # for test set
        candidate_raw_data = ['015', '018', '022', '023', '280', '485', '1404']

        RAW_DATA_PATH = args.raw_path

        pool = multiprocessing.Pool(processes=6)
        pool.map(make_mix_eval_data, permutations(candidate_raw_data, 2))
        pool.close()
        pool.join()
