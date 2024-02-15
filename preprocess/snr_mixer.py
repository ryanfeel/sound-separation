from abc import ABC, abstractmethod
import numpy as np
import soundfile as sf

class Mixer(ABC):
    @abstractmethod
    def mix(clean, noise):
        pass


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
