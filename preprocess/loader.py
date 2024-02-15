from abc import ABC, abstractmethod
import os

from ai.sleep_audio.core import noise_reduce, load


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


class PSGDataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        audio = load(os.path.join(path, 'audio_0.wav'))
        return_audio = list()
        duration = self.sr * 300
        for i in range(0, len(audio), duration):
            return_audio.append(audio[i:i+duration])

        return return_audio, self.sr

    def nr(self, source, sr):
        return noise_reduce(source, sr)


class HomePSGDataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        return_audio = list()
        for file in files:
            audio = load(os.path.join(path, file))
            return_audio.append(audio)

        return return_audio, self.sr

    def nr(self, source, sr):
        return noise_reduce(source, sr)


class ARIADataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        audio = load(os.path.join(path, 'audio_0.wav'))
        if audio.ndim > 1:
            audio = audio[:, 0]
        return_audio = list()
        duration = self.sr * 300
        for i in range(0, len(audio), duration):
            return_audio.append(audio[i:i+duration])

        return return_audio, self.sr

    def nr(self, source, sr):
        return noise_reduce(source, sr)


class ClionicDataLoader(RawDataLoader):
    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        files = os.listdir(path)
        files = sorted(files, key=lambda x: x.split('.')[0])
        return_audio = list()
        for file in files:
            audio = load(os.path.join(path, file))
            duration = self.sr * 300
            for i in range(2):
                start = i * duration
                end = (i+1) * duration
                return_audio.append(audio[start:end])

        return return_audio, self.sr

    def nr(self, source, sr):
        return noise_reduce(source, sr)


class FactoryDataLoader():
    def __init__(self):
        self.PSGLoader = PSGDataLoader()
        self.HomePSGLoader = HomePSGDataLoader()
        self.AriaLoader = ARIADataLoader()
        self.ClionicDataLoader = ClionicDataLoader()

    def loader(self, id):
        id = int(id)
        if id < 100:
            return self.HomePSGLoader
        elif id < 5000:
            return self.PSGLoader
        elif id < 10000:
            return self.ClionicDataLoader
        else:
            return self.AriaLoader
