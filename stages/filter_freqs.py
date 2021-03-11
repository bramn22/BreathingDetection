from stages.stage import StageInterface
import numpy as np
from scipy.fft import fft, fftfreq, rfftfreq, rfft
import matplotlib.pyplot as plt
from scipy import signal
import scipy


class FreqFilter(StageInterface):

    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self.fps = fps

    def _execute(self, inp, **kwargs):
        # Take first 10 seconds!!
        comps = inp[:10*self.fps].T
        # comps = inp.T
        # Zero padding and apply Hamming windowing
        window = signal.get_window('hamming', comps.shape[1])

        # FFT to extract frequency spectrum
        print("Received inp:", inp.shape)
        self.yfs = []
        self.xfs = []
        self.periodicities = []
        n = 50000
        for comp in comps:
            f, psd = signal.welch(x=comp, fs=self.fps, window='hamming', nperseg=256, nfft=1024, scaling='density', average='mean')
            self.yfs.append(psd)
            self.xfs.append(f)

            freq_step_size = f[1] - f[0]
            print("Freq stepsize", freq_step_size)
            asd = np.sqrt(psd)
            # Cutoff all frequencies before 1
            # start_freq_idx = 0
            cutoff_freq = 1
            start_freq_idx = np.argmin(np.abs(f - cutoff_freq))
            asd = asd[start_freq_idx:]
            f = f[start_freq_idx:]
            dom_freq = np.argmax(asd)
            # Find index of frequency that is closest to the first harmonic of the dominant frequency
            harm_freq = np.argmin(np.abs(f - f[dom_freq]*2))
            print(f"Dominant freq: {f[dom_freq]}, Harmonic freq: {f[harm_freq]}")
            # harm_freq = int(dom_freq + (f[dom_freq]*2 - f[dom_freq])//freq_step_size)
            # print(f"Dominant freq: {f[dom_freq]}, Harmonic freq: {f[harm_freq]}")

            window_steps = int(0.05//freq_step_size)
            if window_steps == 0:
                window_steps = 1
            dom_spectral_density = np.sum(asd[dom_freq-window_steps:dom_freq+window_steps+1]) + np.sum(asd[harm_freq-window_steps:harm_freq+window_steps+1])
            total_spectral_density = np.sum(asd)
            periodicity = dom_spectral_density / total_spectral_density
            self.periodicities.append(periodicity)

            # n = len(comp)
            # yf = rfft(comp*window, n=n)
            # self.yfs.append(yf)
            # self.xfs.append(rfftfreq(n, 1 / self.fps))

            # Calculate peak-to-total ratio for all PCs
            # dom_freq = np.argmax(np.abs(yf))
            # # Todo: take bin 0.5 Hz around dom_freq
            # dom_spectral_density = np.sum(np.abs(yf[dom_freq-2:dom_freq+2]))
            # total_spectral_density = np.sum(np.abs(yf))
            # periodicity = dom_spectral_density / total_spectral_density
            # self.periodicities.append(periodicity)
        # Select PC with highest peak-to-total ratio as the breathing rate
        self.top_pcs = np.argsort(self.periodicities)[-8:][::-1]
        return self.top_pcs, inp
        # return self.yfs, self.xfs, self.periodicities

    def _visualize(self):
        fig, axs = plt.subplots(len(self.yfs), 1, figsize=(10, len(self.yfs)*1))
        for comp_i, (xf, yf, per) in enumerate(zip(self.xfs, self.yfs, self.periodicities)):
            # axs[comp_i].plot(xf, 20*scipy.log10(np.abs(yf)))
            axs[comp_i].plot(xf, np.abs(yf))
            axs[comp_i].set_title(f"PC{comp_i}, periodicity={per:.4f}")
        plt.tight_layout()
        plt.show()
