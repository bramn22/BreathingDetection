from stages.stage import StageInterface
import numpy as np
from scipy.fft import fft, fftfreq, rfftfreq, rfft
import matplotlib.pyplot as plt
from scipy import signal
import scipy


class BandpassFilter(StageInterface):

    def __init__(self, fps, cutoffs, **kwargs):
        super().__init__(**kwargs)
        self.fps = fps
        self.cutoffs = cutoffs

    def _execute(self, inp, **kwargs):
        print("Received inp:", inp.shape)
        self.inp = inp
        X = inp.copy()
        # Bandpass filtering
        fil = signal.firwin(numtaps=71, fs=self.fps, cutoff=self.cutoffs, window='blackmanharris', pass_zero='bandpass')

        for feat_i in range(X.shape[1]):
            X[:,feat_i] = signal.lfilter(fil, 1.0, X[:,feat_i])
        self.X = X
        return self.X

    def _visualize(self):
        fig, axs = plt.subplots(5, 2, figsize=(10, 5))
        for ax in axs:
            idx = np.random.randint(0, self.inp.shape[1])
            ax[0].plot(self.inp[:,idx])
            ax[1].plot(self.X[:,idx])
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))

        # Show self-generated signal with freq below in and above cutoff frequencies
        t = 20
        f1, f2, f3 = 0.2, 6, 3
        samples = np.arange(t * self.fps) / self.fps
        s1 = np.sin(2 * np.pi * f1 * samples)
        s2 = np.sin(2 * np.pi * f2 * samples)
        s3 = np.sin(2 * np.pi * f3 * samples)
        total_signal = s1+s2+s3

        # Show magnitude spectrum of signal
        f, psd = signal.welch(x=total_signal, fs=self.fps, window='hamming', nperseg=256, nfft=1024, scaling='density',
                              average='mean')

        # Show filtered signal
        filtered_signal = np.squeeze(self._execute(np.expand_dims(total_signal, -1)))

        # Show magnitude spectrum of filtered signal
        filt_f, filt_psd = signal.welch(x=filtered_signal, fs=self.fps, window='hamming', nperseg=256, nfft=1024, scaling='density',
                              average='mean')

        axs[0, 0].plot(total_signal)
        axs[1, 0].plot(filtered_signal)
        axs[0, 1].plot(f, psd)
        axs[1, 1].plot(filt_f, filt_psd)
        plt.show()