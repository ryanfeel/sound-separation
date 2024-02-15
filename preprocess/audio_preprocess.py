import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

from irr_estimate import _estimate_SNR_with_fft
from ai.sleep_audio.core import audio2mel
from soundsleep.preprocess.utils import mel_spectrogram

# Noise Reduction Functions
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating


def save_waveform(output_path, amp, samplerate):
    sf.write(output_path, amp, samplerate, format="wav")

def drc(signal, threshold_list=[0.1, 0.5, 2.0], minimum_multiple=2.0, maximum_limit=0.95):
    # TODO: delete constant value
    signal_abs = np.abs(signal)
    multiple = np.ones_like(signal)
    multiple = np.where(signal_abs <= threshold_list[2], -0.525 * signal_abs + 1.525, multiple)
    multiple = np.where(signal_abs <= threshold_list[1], -2.5 * signal_abs + 2.25, multiple)
    multiple = np.where(signal_abs <= threshold_list[0], minimum_multiple, multiple)

    result = signal * multiple
    result = np.where(signal_abs > 2.0, maximum_limit, result)
  
    return result

def normalize(signal, max=-1):
    # normalization
    if max == -1:
        max_peak = np.max(np.abs(signal))
    else:
        max_peak = max

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

def noise_reduction_origin(signal, sr):
    # original our noise reduce
    signal = adaptive_noise_reduce(
        signal,
        sr,
        30,
        estimate_noise_method=noise_minimum_energy,
        reduce_noise_method=spectral_gating,
        smoothing=0.0,
    )

    return signal

def get_IRR_SNR(signal, sr):
    mel_spec = audio2mel(signal, apply_preprocessing="")[0]

    SNR_list = []
    for freq_index in range(2, 19):
        series = librosa.power_to_db(mel_spec[freq_index])

        SNR = _estimate_SNR_with_fft(series, 16000, respiration_freq_range=[8/60, 25/60], idx=freq_index)
        SNR_list.append(SNR)
    SNR = np.mean(SNR_list)
    return SNR
