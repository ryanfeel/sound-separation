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
from loader import FactoryDataLoader
from irr_estimate import _estimate_SNR_with_fft
from soundsleep.preprocess.utils import mel_spectrogram

class Mixer(ABC):
    @abstractmethod
    def mix(clean, noise):
        pass

    def save_waveform(output_path, amp, samplerate, subtype):
        sf.write(output_path, amp, samplerate, format="wav", subtype=subtype)


class SNRMixer(Mixer):
    def mix(self, clean, noise, snr):
        assert len(clean) == len(noise), "not equal length"
        
        clean_rms = self.cal_rms(clean)
        noise_rms = self.cal_rms(noise)
        adjusted_noise_rms = self.cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = noise * (adjusted_noise_rms / noise_rms)
        mixed_amp = clean + adjusted_noise_amp
        
        # normalization for avoiding clippling issue
        # print(mixed_amp, mixed_amp.max(axis=0))
        # mixed_amp = mixed_amp / mixed_amp.max(axis=0) * 0.95

        return mixed_amp

    def cal_adjusted_rms(self, clean_rms, snr):
        # snr = 20log(S/N)
        a = float(snr) / 20
        noise_rms = clean_rms / (10 ** a)
        return noise_rms

    def cal_rms(self, amp):
        return np.sqrt(np.mean(np.square(amp), axis=-1))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    args = parser.parse_args()
    return args

def save_waveform(output_path, amp, samplerate):
    sf.write(output_path, amp, samplerate, format="wav")


def preprocess(signal, sr):
    # preprocess before mix

    # normalization
    max_peak = np.max(np.abs(signal))
    if max_peak > 1.0:
        ratio = 1 / max_peak
        signal = signal * ratio * 0.95

    # noise reduce
    signal = nr.reduce_noise(
        y=signal, 
        sr=sr, 
        stationary=False
    )

    return signal

def get_IRR_SNR(audio, sr):
    mel_spec = mel_spec = mel_spectrogram(audio, sr, 20, 50e-3, 25e-3)

    SNR_list = []
    for freq_index in range(2, 19):
        series = librosa.power_to_db(mel_spec[freq_index])

        SNR = _estimate_SNR_with_fft(series, 16000, respiration_freq_range=[8/60, 25/60], idx=freq_index)
        SNR_list.append(SNR)
    SNR = np.mean(SNR_list)
    return SNR

def make_mix_data(raw_data):
    print(raw_data[0], raw_data[1])
    raw_data_1 = raw_data[0]
    raw_data_2 = raw_data[1]
    FactoryDataLoader().loader(int(raw_data_1))
    clean_data, sr = FactoryDataLoader().loader(int(raw_data_1)).load(os.path.join(RAW_DATA_PATH, raw_data_1 + '_data'))
    noise_data, sr = FactoryDataLoader().loader(int(raw_data_2)).load(os.path.join(RAW_DATA_PATH, raw_data_2 + '_data'))
    # length check -> extract method
    if len(clean_data) <= len(noise_data):
        noise_data = noise_data[:len(clean_data)]
    else:
        noise_data.append(noise_data[:(len(clean_data)-len(noise_data))])

    # mix every 30 sec
    j = 0
    for c_audio, n_audio in zip(clean_data[2:-2], noise_data[2:-2]):
        print(j)
        for i in range(10):
            duration = sr * 30                    
            source1 = c_audio[i*duration:(i+1)*duration]
            source2 = n_audio[i*duration:(i+1)*duration]
            
            if get_IRR_SNR(source1, sr) < 10 and get_IRR_SNR(source2, sr) < 10:
                continue

            # preprocess before mix (normalization & noise reduction)
            source1 = preprocess(source1, sr)
            source2 = preprocess(source2, sr)
            snr = random.random() * 12 + 6
            mixture = mixer.mix(source1, source2, snr)
            mixture2 = mixer.mix(source1, source2, 0)
            
            # save mix, clean audio, noise audio
            c_file_name = '_' + str(j) + '_' + str(i)
            output_mix_path = os.path.join(args.output_path, 'mixture', 'mix_'+ raw_data_1 + '_' \
                + raw_data_2 + c_file_name + '_' + str(round(snr, 2)) +'.wav')
            output_mix2_path = os.path.join(args.output_path, 'mixture', 'mix_'+ raw_data_1 + '_' \
                + raw_data_2 + c_file_name + '_0.wav')
            output_source1_path = os.path.join(args.output_path, 'source1', 'source1_' + raw_data_1 + c_file_name + '.wav')
            output_source2_path = os.path.join(args.output_path, 'source2', 'source2_' + raw_data_2 + c_file_name + '.wav')
            save_waveform(output_mix_path, mixture, 16000)
            save_waveform(output_mix2_path, mixture2, 16000)
            save_waveform(output_source1_path, source1, 16000)
            save_waveform(output_source2_path, source2, 16000)
        j += 1

if __name__ == "__main__":
    args = get_args()
    
    '''
        make mixing code
        1. load data -> return audio data splited 5 minutes (Preprocess)
        2. mix (each 30 sec), randomize snr (0~6, 6~12, 12~18) -> make 6 * 3 audio
        3. NR 
        4. save mix, s0, s1 and meta data
    '''

    # select raw data, TODO: will be load from db
    candidate_raw_data = ['001', '003', '010', '016', '017', '028', '029','033', '123', '1324', '1459', '1495']
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
        dataset_name="v0.3"
    )
