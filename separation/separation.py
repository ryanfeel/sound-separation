from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper, get_si_snr
import torchaudio
import torch
import numpy as np
import soundfile as sf
import librosa

# Noise Reduction Functions
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating

model = separator.from_hparams(
    source="/home/ryan/sound-separation/pretrained_models/PHP_v0.2", 
    savedir='/home/ryan/sound-separation/pretrained_models/PHP_v0.2',
    run_opts={"device":"cuda"} 
    )

# TODO: make separator class and overriding sepration method as audio's size
def separation(audio, audio_signal, sr):
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
    loss3 = get_si_snr(audio, est_sources[:, :, :1])
    loss4 = get_si_snr(audio, est_sources[:, :, 1:])

    if loss0 > 0 and loss1 > 0:
        select = -1
    elif loss0 <= loss1:
        select = 0
    else:
        select = 1
    print(loss0, loss1, loss3, loss4, select)
    
    if select == -1:
        est_sources = original_audio
    else:
        est_sources = est_sources[:, :, select].squeeze()
        est_sources = est_sources.detach().cpu().numpy()
        # normalization
        max_peak = np.max(np.abs(est_sources))
        ratio = 1 / max_peak
        est_sources = est_sources * ratio * 0.95
    
    # tensor to np
    return est_sources


audio, sr = torchaudio.load(
    '/data1/ryan/onboarding/onb_audio/HomePSG/result_HomePSG_006_015_snr_18/audio_0.wav'
    )

audio_source1_original, sr = torchaudio.load(
    '/home/ryan/data/onboarding/survey-android/006_data/33_audio.mp3'
)
audio_source2_original, sr = torchaudio.load(
    '/home/ryan/data/onboarding/survey-android/015_data/33_audio.mp3'
)

start = sr * 30 * 3
duration = sr * 30
torchaudio.save("audio_source1.wav", audio_source1_original[:, start:start+duration].detach().cpu(), sr)
torchaudio.save("audio_source2.wav", audio_source2_original[:, start:start+duration].detach().cpu(), sr)

audio_source1, sr = sf.read("audio_source1.wav")
audio_source1 = adaptive_noise_reduce(
    audio_source1,
    sr,
    30,
    estimate_noise_method=noise_minimum_energy,
    reduce_noise_method=spectral_gating,
    smoothing=0.,
)

audio_source2, sr = sf.read("audio_source2.wav")
audio_source2 = adaptive_noise_reduce(
    audio_source2,
    sr,
    30,
    estimate_noise_method=noise_minimum_energy,
    reduce_noise_method=spectral_gating,
    smoothing=0.,
)

sf.write("audio_source1_NR.wav", audio_source1, sr, format="wav")
sf.write("audio_source2_NR.wav", audio_source2, sr, format="wav")

start = sr * 30 * 333
duration = sr * 30
torchaudio.save("audio_mix.wav", audio[:, start:start+duration].detach().cpu(), sr)
audio = audio[:, start:start+duration]

audio, sr = sf.read("audio_mix.wav")
audio = adaptive_noise_reduce(
    audio,
    sr,
    30,
    estimate_noise_method=noise_minimum_energy,
    reduce_noise_method=spectral_gating,
    smoothing=0.,
)


sf.write("audio_mix_NR.wav", audio, sr, format="wav")


audio, sr = torchaudio.load(
    "audio_mix_NR.wav"
)

est_sources_original = model.separate_batch(audio)

print(audio.size(), est_sources_original.size())

est_sources = est_sources_original.detach().cpu().numpy()
max1 = est_sources[:, :, 0].max(axis=-1)
max2 = est_sources[:, :, 1].max(axis=-1)
print(max1, max2)

if max1 > 1:
    est_sources[:, :, 0] = est_sources[:, :, 0] / max1 * 0.95

if max2 > 1:
    est_sources[:, :, 1] = est_sources[:, :, 1] / max2 * 0.95

torchaudio.save("result_s0.wav", torch.Tensor(est_sources[:, :, 0]), sr)
torchaudio.save("result_s1.wav", torch.Tensor(est_sources[:, :, 1]), sr)

audio_source1, sr = torchaudio.load(
    "audio_source1_NR.wav"
)
audio_source2, sr = torchaudio.load(
    "audio_source2_NR.wav"
)
est_sources = torch.Tensor(est_sources)
print(
    get_si_snr(audio_source1.unsqueeze(-1), est_sources[:, :, 1:].to(audio_source1)),
    get_si_snr(audio_source1.unsqueeze(-1), est_sources[:, :, :1].to(audio_source1)),
    get_si_snr(audio_source2.unsqueeze(-1), est_sources[:, :, 1:].to(audio_source1)),
    get_si_snr(audio_source2.unsqueeze(-1), est_sources[:, :, :1].to(audio_source1))
)

