from stages.stage import StageInterface
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class SourceSeparation(StageInterface):

    def __init__(self, n_components, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def _execute(self, inp, meta, **kwargs):
        self.frame_height, self.frame_width = meta['video_height'], meta['video_width']
        print("Received inp:", inp.shape)
        X = inp.copy()
        print(X.shape)
        # pca = PCA(n_components=min(n, height*width))
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

        self.outp = self.pca.transform(X)
        return self.outp

    def _visualize(self):
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.show()

        # Plot component traces
        fig, axs = plt.subplots(self.n_components, 1, figsize=(10, self.n_components*1))
        for comp_i in range(self.n_components):
            axs[comp_i].plot(self.outp[:, comp_i])
            axs[comp_i].set_title(f"PC{comp_i}")
        plt.tight_layout()
        plt.show()

        # Plot eigenfaces
        # fig, axs = plt.subplots(int(np.ceil(self.n_components/4)), 4, figsize=(10, 10))
        # for comp_i in range(self.n_components):
        #     comp_img = np.reshape(self.pca.components_[comp_i], (self.height, self.width))
        #     axs[comp_i//4, comp_i%4].imshow(comp_img, cmap='gray')
        # plt.tight_layout()
        # plt.show()
        fig, axs = plt.subplots(int(np.ceil(self.n_components / 4)), 4, figsize=(10, 10))
        for comp_i in range(self.n_components):
            comp_img = np.reshape(self.pca.components_[comp_i], (self.frame_height, self.frame_width))
            mean_img = np.reshape(self.pca.mean_, (self.frame_height, self.frame_width))
            bound = abs(max(comp_img.min(), comp_img.max(), key=abs))

            # Scale bound by 1/2 so that weights are clearer in the images
            axs[comp_i // 4, comp_i % 4].imshow(comp_img, cmap='bwr', vmin=-1 * bound / 2, vmax=bound / 2, alpha=1)
            axs[comp_i // 4, comp_i % 4].imshow(mean_img, cmap='gray', alpha=0.3)
            axs[comp_i // 4, comp_i % 4].set_title(f"PC{comp_i}")
            axs[comp_i // 4, comp_i % 4].set(xticks=[], yticks=[])
        plt.tight_layout()
        plt.show()


