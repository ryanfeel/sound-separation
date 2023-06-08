from abc import ABC, abstractmethod
import os
import soundfile as sf


class RawDataLoader(ABC):
    @abstractmethod
    def load(self, path):
        '''
        return audio signal splited each 5 minutes
        '''

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


class FactoryDataLoader():
    def __init__(self):
        self.PSGLoader = PSGDataLoader()
        self.HomePSGLoader = HomePSGDataLoader()

    def loader(self, id):
        if id > 100:
            return self.PSGLoader
        else:
            return self.HomePSGLoader

# clean_data = FactoryDataLoader().loader(int('001')).load('/data1/ryan/separation/audio_raw/001_data')
# noise_data = FactoryDataLoader().loader(int('1111')).load('/data1/ryan/separation/audio_raw/1111_data')