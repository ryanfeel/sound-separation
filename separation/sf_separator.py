from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.nnet.losses import get_si_snr
# from CMGAN.src.models import generator

from CMGAN.src.models.generator import TSCNet
from CMGAN.src.evaluation import enhance_one_track, enhance_one_track_with_temp_lowf


import torch
import numpy as np

from preprocess.mixer import remove_low_frequency

model = separator.from_hparams(
    source="/home/ryan/sound-separation/pretrained_models/v0.88", 
    savedir='/home/ryan/sound-separation/pretrained_models/v0.88',
    run_opts={"device":"cuda"} 
    )

def inference_sepformer(audio):
    audio, _, _ = remove_low_frequency(audio)
    audio = torch.Tensor(audio).unsqueeze(0)

    est_sources = model.separate_batch(audio)

    est1 = est_sources[0, :, :1].squeeze(-1)
    est2 = est_sources[0, :, 1:].squeeze(-1)

    return est1.cpu().numpy(), est2.cpu().numpy()


class CMGANV082FFT800():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.82_lr5e4_fft800_hop400/CMGAN_epoch_25_-4.93")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track(
            self.model_c, mixture, self.n_fft, self.n_fft // 2
        )
        est_audio = np.multiply(est_audio, 4.0)
        est_audio2 = np.multiply(est_audio2, 64.0)
        return est_audio, est_audio2


class CMGANV088():
    # 꽤 잘되는 모델 9epoch (without perm)
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.88_after_10_1mix/CMGAN_epoch_9_-4.50")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2, delete_f_bin=8
        )
        est_audio = np.multiply(est_audio, 4.0)
        est_audio2 = np.multiply(est_audio2, 4.0)
        return est_audio, est_audio2


class CMGANV085():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.85_lr5e4_fft800_hop400/CMGAN_epoch_41_-3.40")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track(
            self.model_c, mixture, self.n_fft, self.n_fft // 2
        )
        est_audio = np.multiply(est_audio, 128.0)
        est_audio2 = np.multiply(est_audio2, 128.0)
        return est_audio, est_audio2

class CMGANV082LOWF():
    def __init__(self):
        self.n_fft = 400
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/home/ryan/CMGAN/src/saved_model/v0.82_2est_sisnr_lowf/CMGAN_epoch_6_-5.11")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        # input window size 7 sec, hop size 5 sec
        window_size = 16000 * 15
        hop_size = 16000 * 15
        return_audio = []
        ruturn_audio2 = []
        for i in range(2):
            start = i * hop_size
            end = start + window_size
            n_fft = 400
            
            est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
                self.model_c, mixture[:, start:end], n_fft, n_fft // 4
            )
            
            est_audio = np.multiply(est_audio, 128.0)
            est_audio2 = np.multiply(est_audio2, 64.0)

            return_audio.append(est_audio[:hop_size])
            ruturn_audio2.append(est_audio2[:hop_size])


        return np.concatenate(return_audio), np.concatenate(ruturn_audio2)


class CMGANV086():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.86_lr5e4_fft800_hop400/CMGAN_epoch_42_-3.72")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2
        )
        est_audio = np.multiply(est_audio, 256.0)
        est_audio2 = np.multiply(est_audio2, 4.0)
        return est_audio, est_audio2


class CMGANV0881MIX():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.88_1mix/CMGAN_epoch_51_-5.71")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2
        )
        est_audio = np.multiply(est_audio, 4.0)
        est_audio2 = np.multiply(est_audio2, 4.0)
        return est_audio, est_audio2
    

class CMGANV088AFTER101MIX():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.88_after_10_1mix_mag/CMGAN_epoch_77_-5.79")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2
        )
        est_audio = np.multiply(est_audio, 4.0)
        est_audio2 = np.multiply(est_audio2, 4.0)
        return est_audio, est_audio2


class CMGANV0871():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.87_1_after_10_1mix_mag/CMGAN_epoch_7_-4.82")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2, delete_f_bin=10
        )
        est_audio = np.multiply(est_audio, 4.0)
        est_audio2 = np.multiply(est_audio2, 4.0)
        return est_audio, est_audio2


class CMGANTEST():
    def __init__(self):
        self.n_fft = 800
        self.model_c = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        self.model_c.load_state_dict((torch.load("/nas/ryan/saved_model/v0.88_penalty_mamba/CMGAN_epoch_28_-4.39")))
        self.model_c.eval()

    def inference_cmgan(self, mixture):
        est_audio, est_audio2 = enhance_one_track_with_temp_lowf(
            self.model_c, mixture, self.n_fft, self.n_fft // 2, delete_f_bin=8
        )
        # est_audio = np.multiply(est_audio, 16.0)
        # est_audio2 = np.multiply(est_audio2, 16.0)
        return est_audio, est_audio2


cmgan = CMGANTEST()

# TODO: make separator class and overriding sepration method as audio's size
def separation_nn(mixture, name):
    # TODO: check two person
    # IRR / peak detection
    from sleep_audio import write_audio, respiration_estimate_from_peakdetect, noise_reduce

    mixture = torch.Tensor.numpy(mixture)
    
    pd_value = respiration_estimate_from_peakdetect(mixture)
    # mixture = noise_reduce(mixture)
    result = torch.Tensor(mixture)
    result2 = torch.Tensor(mixture)
    # result = torch.Tensor(mixture)

    if pd_value > 0.4:
        # separate with CMGAN
        audio = result.unsqueeze(0)
        est_source, est_source2 = cmgan.inference_cmgan(audio)
        

        result = torch.Tensor(est_source)
        result2 = torch.Tensor(est_source2)
        
        # separate with Sepformer
        # audio = result
        # est_source, est_source2 = inference_sepformer(audio)
        # result = est_source
        # result2 = est_source2
    write_audio("/home/ryan/data/clionic/" + str(name) + "_raw.wav", mixture)
    write_audio("/home/ryan/data/clionic/" + str(name) + "_est1.wav", result)
    write_audio("/home/ryan/data/clionic/" + str(name) + "_est2.wav", result2)
    result_nr = noise_reduce(result.cpu().numpy())
    write_audio("/home/ryan/data/clionic/" + str(name) + "_est1_nr.wav", result_nr)
    

    return torch.Tensor(result_nr), result2


def separation_nn_all(mixture, num):
    from sleep_audio import write_audio, respiration_estimate_from_peakdetect, noise_reduce
    import time

    mixture = torch.Tensor.numpy(mixture)
    pd_value = respiration_estimate_from_peakdetect(mixture)
    mixture_nr = noise_reduce(mixture)
    result = torch.Tensor(mixture_nr)
    

    file_path = '/home/ryan/data/clionic_carl/' + str(num)
    write_audio(file_path + '.wav', mixture)
    write_audio(file_path + '_nr.wav', mixture_nr)

    # np to tensor
    audio = result.unsqueeze(0)
    
    # separate
    est_sources = model.separate_batch(audio)

    write_audio(file_path + '_s0.wav', est_sources[0, :, :1].squeeze(-1).cpu().numpy())
    write_audio(file_path + '_s1.wav', est_sources[0, :, 1:].squeeze(-1).cpu().numpy())

    # TODO: match permutation
    result = est_sources[0, :, 1:].squeeze(-1)

    return result

def separation_with_signal(audio, audio_signal, sr):
    original_audio = audio

    # np to tensor
    audio = torch.Tensor(audio).unsqueeze(0)
    audio_signal = torch.Tensor(audio_signal).unsqueeze(0)
    
    # separate
    est_sources = model.separate_batch(audio)
    
    audio = audio.unsqueeze(-1).to(est_sources)
    audio_signal = audio_signal.unsqueeze(-1).to(est_sources)
    loss0 = get_si_snr(audio_signal, est_sources[:, :, :1])
    loss1 = get_si_snr(audio_signal, est_sources[:, :, 1:])
    snr_baseline = get_si_snr(audio, audio_signal)

    if loss0 < loss1:
        snr = loss0
        select = 0
    else:
        snr = loss1
        select = 1

    snr_i = snr - snr_baseline
    if snr_i > 0:
        select = -1

    print(loss0, loss1, snr_i, select)
    
    if select == -1:
        est_sources = original_audio
    else:
        est_sources = est_sources[:, :, select].squeeze()
        est_sources = est_sources.detach().cpu().numpy()
        # normalization
        max_peak = np.max(np.abs(est_sources))
        if max_peak > 1.0:
            ratio = 1 / max_peak
            est_sources = est_sources * ratio * 0.95
    
    # tensor to np
    return est_sources
