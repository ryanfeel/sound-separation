import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch
import torchaudio

from soundsleep.preprocess.utils import mel_spectrogram
from ai.sleep_audio import load, write_audio, audio2mel
from CMGAN.src.evaluation import inference


signal_path = '/home/ryan/audio_results/item80503_source2.wav'
signal_path2 = '/home/ryan/audio_results/item80505_source1.wav'
signal = load(signal_path)
signal2 = load(signal_path2)

sr = 16000

def plot_mel_spec(signal, file_path):
    # Mel Spectrogram
    mel_spec = mel_spectrogram(signal, sr, 20, 50e-3, 25e-3)
    fig, ax = plt.subplots()
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(file_path)

def plot_from_mel(mel_spec, file_path):
    fig, ax = plt.subplots()
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(file_path)

root_path = '/home/ryan/audio_results/png/'

model_path = '/home/ryan/CMGAN/src/saved_model/CMGAN_epoch_5_0.127'
noise_path = '/home/ryan/wsc/noise_Env3.wav'
save_path = '/home/ryan/nnnr'

noisy = load(noise_path)
noisy = torch.Tensor(noisy)
import time
start = time.time()
b_noise = torch.reshape(noisy, (-1, 30000))
clean1 = inference(model_path, b_noise)
print(time.time() - start)

# clean = np.concatenate((clean1,clean2,clean3,clean4,clean5,clean6,clean7,clean8,clean9,clean10,clean11,clean12), axis=0)
plot_mel_spec(clean1, root_path + 'nnnr_clean_2.png')

exit()
'''
# write_audio('/home/ryan/wsc/selected/results/deep_noise.wav', signal)
# signal = noise_reduction_origin(signal, sr)
# write_audio('/home/ryan/wsc/selected/results/deep_noise_after_nr.wav', signal)

# # Mel Spectrogram
# mel_spec = mel_spectrogram(signal, sr, 20, 50e-3, 25e-3)
# fig, ax = plt.subplots()
# mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
# ax.set(title='Mel spectrogram display')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# plt.savefig('/home/ryan/wsc/selected/results/deep_noise_after_nr.png')


# Mel Spectrogram
mel_spec = mel_spectrogram(signal2, sr, 20, 50e-3, 25e-3)
fig, ax = plt.subplots()
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel spectrogram display')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig('/home/ryan/wsc/selected/results/deep_dryer.png')

# Mel Spectrogram
mel_spec = mel_spectrogram(signal3, sr, 20, 50e-3, 25e-3)
fig, ax = plt.subplots()
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel spectrogram display')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig('/home/ryan/wsc/selected/results/deep_vehicle.png')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

window_size = 50e-3
stride = 25e-3
n_fft = int(sr * window_size)
hop_length = int(sr * stride)

tg = TG(
    sr=16000, 
    nonstationary=True,
    n_thresh_nonstationary=-2.0,
    hop_length=hop_length,
    n_fft=n_fft,
    temp_coeff_nonstationary=10,
    prop_decrease=0.95
).to(device)


noise_reduced_sig = nr.reduce_noise(
            y=signal, 
            sr=sr, 
            stationary=True
        )

# noise_reduced_sig = noise_reduction_origin(signal, 16000)

signal = torch.Tensor(signal).unsqueeze(0).to(device)
noise_reduced_sig = tg(signal)
noise_reduced_sig = noise_reduced_sig.cpu().detach().numpy()
print(noise_reduced_sig)
write_audio('/home/ryan/wsc/selected/deep_noise_after_tgnr.wav', noise_reduced_sig[0])




'''

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sr = 16000

# signal[:48000] = 0.0
window_size = 50e-3
stride = 25e-3
n_fft = int(sr * window_size)
hop_length = int(sr * stride)

sg = StreamedTorchGate(
    y=signal,
    sr=sr,
    stationary=True,
    hop_length=hop_length,
    n_fft=n_fft,
    time_constant_s=20,
    sigmoid_slope_nonstationary=10,
    thresh_n_mult_nonstationary=2.0,
    device=device
)

sg2 = StreamedTorchGate(
    y=signal,
    sr=sr,
    stationary=True,
    thresh_n_mult_nonstationary=0.8,
    device=device
)

sg3 = StreamedTorchGate(
    y=signal,
    sr=sr,
    stationary=False,
    device=device
)

noise_reduced_sig_gpu = sg.get_traces()


# signal_gpu = torch.Tensor(signal).unsqueeze(0).to(device)
# signal_gpu = torch.where(signal_gpu == 0.0, torch.tensor(1e-20, dtype=signal_gpu.dtype).to(device), signal_gpu)
# noise_reduced_sig_gpu = tg(signal_gpu)

print("gpu noisereduce: ", noise_reduced_sig_gpu)

noise_reduced_sig_gpu = np.nan_to_num(noise_reduced_sig_gpu, nan=0.0)

# noise_reduced_sig_gpu = noise_reduced_sig_gpu.cpu().detach().numpy()[0]
# noise_reduced_sig_gpu = noise_reduced_sig_gpu[0].astype('float32')

        
# Mel Spectrogram
mel_spec = mel_spectrogram(noise_reduced_sig_gpu, sr, 20, 50e-3, 25e-3)
fig, ax = plt.subplots()
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel spectrogram display')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig('/home/ryan/wsc/selected/deep_noise_after_noisereduce_r5.png')

def make_mel(path):
    npy_list = glob.glob(path + '/*')
    npy_list.sort()
    
    i = 0
    for path in npy_list:
        # print(path)
        mel_spec = np.load(path, allow_pickle=True).item()
        # print(mel_spec['x'])
        
        for data in mel_spec['x']:
            for d in data:
                if d == 0:
                    print("zero")
                elif np.isnan(d):
                    print("nan")
        '''
        fig, ax = plt.subplots()
        mel_spec = librosa.power_to_db(mel_spec['x'], ref=np.max)

        img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig('/home/ryan/sound-separation/preprocess/test_npy/new_' + str(i) + '.png')
        i += 1
        '''

# make_mel('/home/ryan/sound-separation/preprocess/test_npy/u1232_s1232')



def make_mel(path):
    npy_list = glob.glob(path + '/*')
    npy_list.sort()
    
    i = 0
    for path in npy_list:
        mel_spec = np.load(path, allow_pickle=True).item()
 
        fig, ax = plt.subplots()
        mel_spec = librosa.power_to_db(mel_spec['x'], ref=np.max)

        img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig('/home/ryan/sound-separation/preprocess/test_npy/new_' + str(i) + '.png')
        i += 1