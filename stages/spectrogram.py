from stages.stage import StageInterface
import numpy as np
from scipy.fft import fft, fftfreq, rfftfreq, rfft
import matplotlib.pyplot as plt
from scipy import signal
import scipy


class Spectrogram(StageInterface):

    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self.fps = fps

    def _execute(self, inp, **kwargs):
        top_pcs, comps_T = inp
        comps = comps_T.T
        for pc, comp in zip(top_pcs, comps):
            freqs, times, spectrogram = signal.spectrogram(x=comp, fs=self.fps, nperseg=128, nfft=1024, scaling='density', mode='psd')
            spectrogram_normed = spectrogram / np.max(spectrogram, axis=0)
            freqs = freqs[:300]
            spectrogram_normed = spectrogram_normed[:300]
            print("Times:", times)
            fig = plt.figure(figsize=(5, 4))
            im = plt.pcolormesh(times, freqs, spectrogram_normed, shading='gouraud', vmax=1)
            fig.colorbar(im)

            # plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
            plt.title(f'Spectrogram-PC{pc}')
            plt.ylabel('Frequency band')
            plt.xlabel('Time window')
            plt.tight_layout()
            plt.savefig(f'Spectrogram-PC{pc}.png', dpi=150)
            plt.show()
        return comps


    def _visualize(self):
        pass
        # fig, axs = plt.subplots(len(self.yfs), 1, figsize=(10, len(self.yfs)*1))
        # for comp_i, (xf, yf, per) in enumerate(zip(self.xfs, self.yfs, self.periodicities)):
        #     # axs[comp_i].plot(xf, 20*scipy.log10(np.abs(yf)))
        #     axs[comp_i].plot(xf, np.abs(yf))
        #     axs[comp_i].set_title(f"PC{comp_i}, periodicity={per:.4f}")
        # plt.tight_layout()
        # plt.show()
