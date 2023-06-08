from mixer import SNRMixer
import soundfile as sf
import librosa
import numpy as np
import glob
import os
import random

from irr_measurement.irr_estimate import _estimate_SNR_with_fft
from soundsleep.preprocess.utils import mel_spectrogram
from mixer import save_waveform, create_custom_dataset

def get_IRR_SNR(audio, sr):
    mel_spec = mel_spec = mel_spectrogram(audio, sr, 20, 50e-3, 25e-3)

    SNR_list = []
    for freq_index in range(2, 19):
        series = librosa.power_to_db(mel_spec[freq_index])

        SNR = _estimate_SNR_with_fft(series, 16000, respiration_freq_range=[8/60, 25/60], idx=freq_index)
        SNR_list.append(SNR)
    SNR = np.mean(SNR_list)
    return SNR

RAW_DATA_PATH = '/data1/ryan/separation/v1'
OUTPUT_DATA_PATH = '/data1/ryan/separation/v0.2'
mixer = SNRMixer()

csv_list = list()

file_list = glob.glob('/data1/ryan/separation/v1/mixture/*')

for file in file_list:
    file_name = '_' + file.split('_')[3] + '_' + file.split('_')[4] + '.wav'
    source1_name = 'source1_' + file.split('_')[1] + file_name
    source2_name = 'source2_' + file.split('_')[2] + file_name
    source1, sr = sf.read(os.path.join(RAW_DATA_PATH, 'source1', source1_name))
    source2, sr = sf.read(os.path.join(RAW_DATA_PATH, 'source2', source2_name))
    
    snr = random.randrange(12, 19)
    if get_IRR_SNR(source1, sr) > 10 and get_IRR_SNR(source2, sr) > 10:
        mixture = mixer.mix(source1, source2, snr)
        save_waveform(os.path.join(OUTPUT_DATA_PATH, 'mixture', file.split('/')[-1]), mixture, 16000)
        save_waveform(os.path.join(OUTPUT_DATA_PATH, 'source1', source1_name), source1, 16000)
        save_waveform(os.path.join(OUTPUT_DATA_PATH, 'source2', source2_name), source2, 16000)

# write meta data
create_custom_dataset(
    OUTPUT_DATA_PATH,
    OUTPUT_DATA_PATH,
    dataset_name="v0.2"
)
