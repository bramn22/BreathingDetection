import os
from stages.load_video import VideoLoader
from stages.source_separation import SourceSeparation
from stages.filter_freqs import FreqFilter
from stages.bandpass_filtering import BandpassFilter
from stages.reshaper import Reshape
from stages.spectrogram import Spectrogram
from pipeline import Pipeline

data_path = r'videos'
video_paths = [os.path.join(data_path, video) for video in os.listdir(data_path)]
# video_width, video_height = 644, 484#644, 484
meta = {'video_width': 644, 'video_height': 484}

# Load video
video_loader = VideoLoader(concat_output=True, visualize=True)#, crop=(240, 320, 160, 80))
bandpass_filtering = BandpassFilter(fps=50, cutoffs=[1, 5], visualize=True)
source_separation = SourceSeparation(n_components=20, visualize=True)
freq_filter = FreqFilter(fps=50, visualize=True)
spectogram = Spectrogram(fps=50)

line = Pipeline()
line.add_stage(video_loader)
line.add_stage(Reshape(outp_shape=(-1, meta['video_width']*meta['video_height'])))
line.add_stage(bandpass_filtering)
line.add_stage(source_separation)
line.add_stage(freq_filter)
line.add_stage(spectogram)
line.execute(video_paths, meta)
# Crop video

# Source separation

# Filter sources

# Visualize remaining sources