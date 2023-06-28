import argparse
import numpy as np
import soundfile as sf
from abc import ABC, abstractmethod
import os
import random
import librosa
from itertools import permutations
import multiprocessing

# Noise Reduction Functions
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating
import noisereduce as nr

from util.meta import create_custom_dataset
from preprocess.loader import FactoryDataLoader
from irr_estimate import _estimate_SNR_with_fft
from soundsleep.preprocess.utils import mel_spectrogram

class Mixer(ABC):
    @abstractmethod
    def mix(clean, noise):
        pass

    def save_waveform(output_path, amp, samplerate, subtype):
        sf.write(output_path, amp, samplerate, format="wav", subtype=subtype)


class SNRMixer(Mixer):
    def mix(self, clean, noise, snr, sr):
        assert len(clean) == len(noise), "not equal length"
        
        noise_ratio = 1.0

        if snr != 0:
            noise_ratio = self.cal_noise_ratio(snr)

        adjusted_noise_amp = noise * noise_ratio
        mixed_amp = clean + adjusted_noise_amp
        
        return mixed_amp

    def cal_adjusted_rms(self, clean_rms, snr):
        # snr = 20log(S/N)
        a = float(snr) / 20
        noise_rms = clean_rms / (10 ** a)
        return noise_rms

    def cal_rms(self, amp):
        return np.sqrt(np.mean(np.square(amp), axis=-1))
    
    def cal_noise_ratio(self, snr):
        a = float(snr) / 20
        noise_ratio = 10 ** a
        
        return 1/noise_ratio

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dataset_name", type=str, default="v0.5")
    args = parser.parse_args()
    return args

def save_waveform(output_path, amp, samplerate):
    sf.write(output_path, amp, samplerate, format="wav")


def normalize(signal, sr):
    # normalization
    max_peak = np.max(np.abs(signal))
    if max_peak >= 1.0:
        ratio = 1 / max_peak
        signal = signal * ratio * 0.5

    return signal

def preprocess(signal, sr):
    # preprocess before mix
    # noise reduce
    signal = nr.reduce_noise(
        y=signal, 
        sr=sr, 
        stationary=True
    )

    return signal

def get_IRR_SNR(audio, sr):
    mel_spec = mel_spectrogram(audio, sr, 20, 50e-3, 25e-3)

    SNR_list = []
    for freq_index in range(2, 19):
        series = librosa.power_to_db(mel_spec[freq_index])

        SNR = _estimate_SNR_with_fft(series, 16000, respiration_freq_range=[8/60, 25/60], idx=freq_index)
        SNR_list.append(SNR)
    SNR = np.mean(SNR_list)
    return SNR

def make_mix_eval_data(raw_data):
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
    if len(clean_data) <= len(noise_data):
        noise_data = noise_data[:len(clean_data)]
    else:
        noise_data.append(noise_data[:(len(clean_data)-len(noise_data))])

    # mix every 30 sec
    j = 0
    mix_snr_list = [[], [], [], []]
    output_mix_path = []

    for c_audio, n_audio in zip(clean_data, noise_data):
        c_audio = normalize(c_audio, sr)
        n_audio = normalize(n_audio, sr)
        
        for i in range(10):
            duration = sr * 30
            if len(c_audio) < i*duration:
                continue
                
            source1 = c_audio[i*duration:(i+1)*duration]
            source2 = n_audio[i*duration:(i+1)*duration]
            
            if len(source1) < len(source2):
                source2 = source2[:len(source1)]
            
            source1 = preprocess(source1, sr)
            source2 = preprocess(source2, sr)

            for j in range(4):
                mix_snr_list[j] = np.concatenate([mix_snr_list[j], mixer.mix(source1, source2, j*6, sr)])

    # save mix
    for i in range(4):
        if int(raw_data_1) < 100:
            data_type = 'HomePSG'
        else:
            data_type = 'PSG'
        path_name = os.path.join(
            args.output_path,
            'result_' + data_type + '_' + raw_data_1 + '_' + raw_data_2 + '_snr_' + str(i*6)
        )
        if os.path.isdir(path_name) is not True:
            os.mkdir(path_name)
        output_mix_path.append(path_name)

        save_waveform(os.path.join(output_mix_path[i], 'audio_0.wav'), mix_snr_list[i], 16000)

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
    for c_audio, n_audio in zip(clean_data[2:-2], noise_data[2:-2]):
        c_audio = normalize(c_audio, sr)
        n_audio = normalize(n_audio, sr)
        for i in range(10):
            duration = sr * 30                    
            source1 = c_audio[i*duration:(i+1)*duration]
            source2 = n_audio[i*duration:(i+1)*duration]
            
            source1 = preprocess(source1, sr)
            source2 = preprocess(source2, sr)
            
            if get_IRR_SNR(source1, sr) < 15 or get_IRR_SNR(source2, sr) < 15:
                continue

            # get and save mixture
            c_file_name = '_' + str(j) + '_' + str(i)
            for j in range(3):
                snr = random.random() * 6 + j * 6
                mixture = mixer.mix(source1, source2, snr, sr)
                output_mix_path = os.path.join(
                    args.output_path, 
                    'mixture', 
                    'mix_'+ raw_data_1 + '_' + raw_data_2 +\
                        c_file_name + '_' + str(round(snr, 2)) +'.wav')
                save_waveform(output_mix_path, mixture, 16000)

            # save clean and noise audio
            output_source1_path = os.path.join(args.output_path, 'source1', 'source1_' + raw_data_1 + c_file_name + '.wav')
            output_source2_path = os.path.join(args.output_path, 'source2', 'source2_' + raw_data_2 + c_file_name + '.wav')
            save_waveform(output_source1_path, source1, 16000)
            save_waveform(output_source2_path, source2, 16000)
            # TODO: SNR 비에 따라서 source2도 다른 크기로 저장

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
        candidate_raw_data = ['003', '010', '016', '017', '029', '033', '123', '1324', '1459', '1495']
        RAW_DATA_PATH = args.raw_path
        mixer = SNRMixer()

        pool = multiprocessing.Pool(processes=8)
        pool.map(make_mix_data, permutations(candidate_raw_data, 2))
        pool.close()
        pool.join()

        # write meta data
        create_custom_dataset(
            args.output_path,
            args.output_path,
            args.dataset_name
        )

    elif args.mode == 'eval':
        # for test set
        candidate_raw_data = ['015', '018', '022', '023', '280', '485', '1404']

        RAW_DATA_PATH = args.raw_path
        mixer = SNRMixer()

        pool = multiprocessing.Pool(processes=6)
        pool.map(make_mix_eval_data, permutations(candidate_raw_data, 2))
        pool.close()
        pool.join()
