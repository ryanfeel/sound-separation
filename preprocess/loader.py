from abc import ABC, abstractmethod
import os
import soundfile as sf
import librosa
from preprocess.audio_preprocess import save_waveform, normalize, \
    noise_reduction, noise_reduction_origin, get_IRR_SNR


class RawDataLoader(ABC):
    @abstractmethod
    def load(self, path):
        '''
        return audio signal splited each 5 minutes
        '''

    @abstractmethod
    def nr(self):
        '''
        return noise reduction method
        '''

    # dereverb (loader마다 dereverb 혹은 변수 놓기)
    # clionic 데이터로부터 방 환경 확인 후 dereverb test 

class PSGDataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        audio, sr = sf.read(os.path.join(path, 'audio_0.wav'))
        return_audio = list()
        duration = self.sr * 300
        for i in range(0, len(audio), duration):
            return_audio.append(audio[i:i+duration])

        return return_audio, sr

    def nr(self, source, sr):
        return noise_reduction(source, sr)


class HomePSGDataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        return_audio = list()
        for file in files:
            audio, sr = sf.read(os.path.join(path, file))
            return_audio.append(audio)

        return return_audio, sr

    def nr(self, source, sr):
        return noise_reduction(source, sr)


class ARIADataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        audio, sr = sf.read(os.path.join(path, 'audio_0.wav'))
        return_audio = list()
        duration = self.sr * 300
        for i in range(0, len(audio), duration):
            return_audio.append(audio[i:i+duration])

        return return_audio, sr

    def nr(self, source, sr):
        return noise_reduction_origin(source, sr)


class FactoryDataLoader():
    def __init__(self):
        self.PSGLoader = PSGDataLoader()
        self.HomePSGLoader = HomePSGDataLoader()
        self.AriaLoader = ARIADataLoader()

    def loader(self, id):
        id = int(id)
        if id < 100:
            return self.HomePSGLoader
        elif id < 10000:
            return self.PSGLoader
        else:
            return self.AriaLoader
