import argparse
import os
import torch
from functools import partial
import multiprocessing
import glob
import random

from sleep_audio import load, write_audio, noise_reduce
# from sleep_audio.core import power_compress, power_uncompress, power_compress_with_low_f_delete, power_uncompress_with_low_f_delete
from preprocess.room_simulation import RoomSimulator
from CMGAN.src.utils import power_compress, power_compress_with_low_f_delete, power_uncompress_with_low_f_delete, power_uncompress


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    args = parser.parse_args()
    return args

def check_again_select(session_id1, session_id2):
    return_value = False
    if session_id1 == session_id2:
        return_value = True
    elif session_id1 < 100 and session_id2 > 100:
        return_value = True
    elif session_id1 > 100 and session_id2 < 100:
        return_value = True

    # print(session_id1, session_id2, return_value)
    return return_value
    
    
def select_candidate(raw_path):
    sample_list = glob.glob(raw_path + '/*')

    s_path = random.choice(sample_list)
    s_session_id = s_path.split('_')[-2]
    s_index = s_path.split('_')[-1]
    
    n_path = random.choice(sample_list)
    n_session_id = n_path.split('_')[-2]
    n_index = n_path.split('_')[-1]

    
    while check_again_select(int(s_session_id), int(n_session_id)):
        n_path = random.choice(sample_list)
        n_session_id = n_path.split('_')[-2]
        n_index = n_path.split('_')[-1]

    return s_path, s_session_id, s_index, n_path, n_session_id, n_index

def remove_low_frequency(audio, n_fft=400, hop=100):
    clean = torch.Tensor(audio).unsqueeze(0)

    clean_spec = torch.stft(
        clean,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft),
        onesided=True,
        return_complex=True
    )
    clean_spec, temp_mag, temp_phase = power_compress_with_low_f_delete(clean_spec, delete_f_bin=8)

    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
    
    clean_uncompress = power_uncompress(clean_real, clean_imag).squeeze(1)

    clean_audio = torch.istft(
        clean_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft),
        onesided=True,
    )
    clean_audio = clean_audio / 4.0

    return clean_audio.squeeze(0).cpu().numpy(), temp_mag, temp_phase

def restore_low_frequency(audio, temp_mag, temp_phase, n_fft=400, hop=100):
    clean = torch.Tensor(audio).unsqueeze(0)

    clean_spec = torch.stft(
        clean,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft),
        onesided=True,
        return_complex=True
    )
    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
    
    clean_uncompress = power_uncompress_with_low_f_delete(clean_real, clean_imag, temp_mag, temp_phase, delete_f_bin=8).squeeze(1)
    clean_audio = torch.istft(
        clean_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft),
        onesided=True,
    )
    clean_audio = clean_audio / 4.0

    return clean_audio.squeeze(0).cpu().numpy()

def make_mix_data_from_sample(process_num, raw_path, save_path, make_num, sr=16000):
    print(process_num, raw_path, save_path, make_num)
    x = process_num * make_num
    for i in range(make_num):
        if i % 100 == 0:
            mixer = RoomSimulator()

        s_path, s_session_id, s_index, n_path, n_session_id, n_index  =\
            select_candidate(raw_path)

        source1 = load(s_path)
        source2 = load(n_path)
        
        # source1 = noise_reduce(source1, method="default_np")
        # source2 = noise_reduce(source2, method="default_np")

        # source1, s1_temp_mag, s1_temp_phase = remove_low_frequency(source1, n_fft=800, hop=400)
        source2, s2_temp_mag, s2_temp_phase = remove_low_frequency(source2, n_fft=800, hop=400)


        c_file_name = '_' + s_index.split('.')[0] + '_' + n_index.split('.')[0]

        output_mix_path = os.path.join(
            save_path, 'mixture', 
            'mix_' + str(x) + '_' + s_session_id + '_' + n_session_id +\
                c_file_name + '.wav')
        output_source1_path = os.path.join(
            save_path, 'source1', 
            'source1_' + str(x) + '_' + s_session_id + '_' + s_index.split('.')[0] + '.wav')        
        output_source2_path = os.path.join(
            save_path, 'source2', 
            'source2_' + str(x) + '_' + n_session_id + '_' + n_index.split('.')[0] + '.wav')

        _, source2 = mixer.simulate(source1, source2)
        
        source1 = source1[:30*16000]
        source2 = source2[:30*16000]

        # for restoring deleted low freqeuncy
        # source1 = restore_low_frequency(source1, s1_temp_mag, s1_temp_phase, n_fft=800, hop=400)
        # source2 = restore_low_frequency(source2, s2_temp_mag, s2_temp_phase, n_fft=800, hop=400)

        mixture = source1 + source2
        mixture = mixture[:30*16000]

        write_audio(output_source1_path, source1, sr)
        write_audio(output_source2_path, source2, sr)
        write_audio(output_mix_path, mixture, sr)
        x = x + 1


if __name__ == "__main__":
    args = get_args()
    
    '''
        make mixing code
        1. load source data (30 sec) 
        2. preprocess(optional): noise reduction
        3. mix two sources varied ways (just add, room simulator, ...)
        4. save mix, s0, s1 and meta data
    '''

    split_path = ['train', 'test']
    source_path = ['mixture', 'source1', 'source2']

    if os.path.isdir(args.output_path) is not True:
        os.mkdir(args.output_path)
        for split in split_path:
            os.mkdir(os.path.join(args.output_path, split))
            for source in source_path:
                os.mkdir(os.path.join(args.output_path, split, source))

    process_num = 4
    make_num = 5000
    # make_num = 5
    for split in split_path:
        if split != 'train':
            make_num = 500
            # make_num = 1
        raw_path = os.path.join(args.raw_path, split)
        save_path = os.path.join(args.output_path, split)
        pool = multiprocessing.Pool(processes=process_num)
        pool.map(
            partial(make_mix_data_from_sample, raw_path=raw_path, save_path=save_path, make_num=make_num), 
            range(0, process_num)
        )
        pool.close()
        pool.join()
    # make_mix_data_from_sample(raw_path, 100, 0)
