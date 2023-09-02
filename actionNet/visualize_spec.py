import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import pandas as pd
from scipy import signal
import librosa
import matplotlib.pyplot as plt

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show(block=False)

# Sampling frequency is 160 Hz
# With 32 samples the frequency resolution after FFT is 160 / 32 = 5Hz: not good we have filtered to 5Hz before
# with 

n_fft = 32
win_length = None
hop_length = 4

spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True
)


def audio_Spectrogram(wave, wav_name, fs, plot=False):
    '''

    Parameters
    ----------
    wave : wav file, SINGLE AUDIO FILE.
    wav_name : string, AUDIO FILE NAME (necessary when plot is True)
    fs : int, SAMPLE FREQUENCY .
    plot : bool. The default is False.

    Returns
    -------
    f : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    Sxx : TYPE
        DESCRIPTION.

    '''

    # SETTING SUGGESTED VALUES FOR SPECTROGRAM BOOK MACHINE LEARNING..12.3.2 The Speech Front End
    deltat = int(0.02*fs)  # overlapping of 20ms
    width = int(0.03*fs)  # window of 30 ms book -> for fs=160 --> 5 samples: Fresol=fs/width = 160/5=32 Hz
    
    deltat = int(0.012*fs)  # overlapping of 20ms
    width = int(0.02*fs)
    #deltat = int(0.12*fs)  # overlapping of 120ms
    #width=int(0.18*fs) # 20 ms ->    ;widthof 180 ms, --> = 28.8 samples -> fres= 160/28.8 = 5.5


    w = signal.windows.hamming(width)
    f, t, Sxx = signal.spectrogram(wave, fs, window=w, noverlap=deltat)
    if plot:
        fig3, ax3 = plt.subplots()
        im = ax3.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        fig3.colorbar(im, ax=ax3)
        ax3.set_ylabel('Frequency [Hz]')
        # ax3.set( yscale="log")
        ax3.set_xlabel('Time [sec]')
        ax3.set_title(wav_name)
        return fig3, ax3
    else:
        return f, t, Sxx

def compute_spectrogram(signal, title):
    # freq_signal = [spectrogram(signal[:, i]) for i in range(8)]
    freq_signal=[]
    Sxx=[]
    Sxx_=[]
    for i in range(8):
        tmp=spectrogram(torch.from_numpy(signal[:,i]))
        f,t,tmp2=audio_Spectrogram(signal[:,i],"",160,False)
        tmp2=torch.from_numpy(tmp2)
        tmp2_=torch.nn.functional.normalize(tmp2)#,dim=1)
        Sxx.append(tmp2)
        Sxx_.append(tmp2_)
        freq_signal.append(tmp)

    #plot_spectrogram(freq_signal, title=title)
    #plot_spectrogram(Sxx, title="Sxx")

    #return torch.stack(freq_signal)
    if(title=="LSTM"):
        return torch.stack(freq_signal)
    else:
        a=torch.stack(Sxx_).view(1,-1,Sxx_[0].size(1)) #[shape]: [1,16,49]
        return a
    #return torch.stack(Sxx_)# [8,2,49]
    
# Replace with your path to one of the subjects from Action-Net
# emg_annotations = pd.read_pickle("/Users/cesaraugustoseminarioyrigoyen/Documents/CORSI/DATA_SCIENCE_POLI/II_MLDL/Project/Project1A_git/mldl23-ego/action-net/ActionNet_train.pkl")
# # emg_annotations = pd.read_pickle("../../aml22-ego-solutions/action-net/emg_annotations/S04_1.pkl")
# sample_no = 1
# signal = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
# # signal = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()

# title = emg_annotations.iloc[sample_no].description
# compute_spectrogram(signal, title)