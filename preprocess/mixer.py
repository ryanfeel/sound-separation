import argparse
import numpy as np
import soundfile as sf
from abc import ABC, abstractmethod
import os
import random
import csv

# Noise Reduction Functions
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating

from loader import FactoryDataLoader

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


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="v0.1"
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    mix_path = os.path.join(datapath, "mixture")
    s1_path = os.path.join(datapath, "source1")
    s2_path = os.path.join(datapath, "source2")
    files = os.listdir(mix_path)

    mix_fl_paths = list()
    s1_fl_paths = list()
    s2_fl_paths = list()

    for fl in files:
        mix_fl_paths.append(os.path.join(mix_path, fl))
        
        s1 = fl.split('_')[1]
        s2 = fl.split('_')[2]
        id = '_' + fl.split('_')[3] + '_' + fl.split('_')[4]
        s1_fl_paths.append(os.path.join(s1_path, 'source1_' + s1 + id + '.wav'))
        s2_fl_paths.append(os.path.join(s2_path, 'source2_' + s2 + id + '.wav'))

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
    ]

    with open(
        os.path.join(savepath, dataset_name + "_train.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (mix_path, s1_path, s2_path) in enumerate(
            zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
        ):

            row = {
                "ID": i,
                "duration": 1.0,
                "mix_wav": mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_path,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_path,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
            }
            writer.writerow(row)

if __name__ == "__main__":
    args = get_args()
    # random.seed()
    
    '''
        make mixing code
        1. load data -> return audio data splited 5 minutes (Preprocess)
        2. mix (each 30 sec), randomize snr (0~6, 6~12, 12~18) -> make 6 * 3 audio
        3. NR 
        4. save mix, s0, s1 and meta data
    '''

    # select raw data, TODO: will be load from db
    candidate_raw_data = ['001', '003', '010', '016', '017', '123', '1324', '1459', '1495']
    RAW_DATA_PATH = args.raw_path

    mixer = SNRMixer()

    csv_list = list()

    # TODO: use multi thread 
    for raw_data_1 in candidate_raw_data:
        for raw_data_2 in candidate_raw_data:
            if raw_data_1 == raw_data_2:
                continue
            print(raw_data_1, raw_data_2)
            clean_data = FactoryDataLoader().loader(int(raw_data_1)).load(os.path.join(RAW_DATA_PATH, raw_data_1 + '_data'))
            noise_data = FactoryDataLoader().loader(int(raw_data_2)).load(os.path.join(RAW_DATA_PATH, raw_data_2 + '_data'))
            # length check -> extract method
            if len(clean_data) <= len(noise_data):
                noise_data = noise_data[:len(clean_data)]
            else:
                noise_data.append(noise_data[:(len(clean_data)-len(noise_data))])

            # mix every 30 sec
            j = 0
            for c_audio, n_audio in zip(clean_data[2:-2], noise_data[2:-2]):
                print(j)
                snr = random.random() * 18 + 3
                for i in range(6):
                    duration = 16000 * 30                    
                    source1 = c_audio[i*duration:(i+1)*duration]
                    source2 = n_audio[i*duration:(i+1)*duration]
                    
                    # Noise Reduction
                    source1 = adaptive_noise_reduce(
                        source1,
                        16000,
                        30,
                        estimate_noise_method=noise_minimum_energy,
                        reduce_noise_method=spectral_gating,
                        smoothing=0.,
                    )
                    source2 = adaptive_noise_reduce(
                        source2,
                        16000,
                        30,
                        estimate_noise_method=noise_minimum_energy,
                        reduce_noise_method=spectral_gating,
                        smoothing=0.,
                    )
                    
                    mixture = mixer.mix(source1, source2, snr)

                    # clipping 
                    
                    # save mix, clean audio, noise audio
                    c_file_name = '_' + str(j) + '_' + str(i)
                    output_mix_path = os.path.join(args.output_path, 'mix_'+ raw_data_1 + '_' + raw_data_2 + c_file_name + '_' + str(round(snr, 2)) +'.wav')
                    output_source1_path = os.path.join(args.output_path, 'source1_' + raw_data_1 + c_file_name + '.wav')
                    output_source2_path = os.path.join(args.output_path, 'source2_' + raw_data_2 + c_file_name + '.wav')
                    save_waveform(output_mix_path, mixture, 16000)
                    save_waveform(output_source1_path, source1, 16000)
                    save_waveform(output_source2_path, source2, 16000)
                    csv_list.append([output_mix_path, output_source1_path, output_source2_path])
                j += 1

    # write meta data
    create_custom_dataset(
        args.output_path,
        args.output_path,
        dataset_name="v0.1"
    )
