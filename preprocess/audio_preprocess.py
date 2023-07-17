import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

from irr_estimate import _estimate_SNR_with_fft
from soundsleep.preprocess.utils import mel_spectrogram

# Noise Reduction Functions
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating

def save_waveform(output_path, amp, samplerate):
    sf.write(output_path, amp, samplerate, format="wav")

def normalize(signal):
    # normalization
    max_peak = np.max(np.abs(signal))
    if max_peak >= 1.0:
        ratio = 1 / max_peak
        signal = signal * ratio * 0.5

    return signal

def noise_reduction(signal, sr):
    # noise reduce
    signal = nr.reduce_noise(
        y=signal, 
        sr=sr, 
        stationary=True
    )

    return signal

def noise_reduction_origin(source, sr):
    # original our noise reduce
    signal = adaptive_noise_reduce(
        source,
        sr,
        30,
        estimate_noise_method=noise_minimum_energy,
        reduce_noise_method=spectral_gating,
        smoothing=0.0,
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