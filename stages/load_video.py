from stages.stage import StageInterface
import numpy as np
import cv2
import matplotlib.pyplot as plt


class VideoLoader(StageInterface):

    def __init__(self, crop=None, **kwargs):
        super().__init__(**kwargs)
        # crop is either None, or (x, y, width, height) with x and y being the coordinates for the top left corner
        self.crop = crop
        self.first_frame = None

    def _execute(self, inp, meta, **kwargs):
        videos = []
        self.height = -1
        self.width = -1
        self.frame_count = 0
        for video_path in inp:
            print("Loading video: ", video_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_count += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.crop is not None:
                video = np.zeros(shape=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.crop[3], self.crop[2]))
            else:
                video = np.zeros(shape=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.height, self.width))

            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.squeeze(frame)
                    if self.first_frame is None:
                        self.first_frame = frame.copy()

                    if self.crop is not None:
                        x, y, w, h = self.crop[0], self.crop[1], self.crop[2], self.crop[3]
                        frame = frame[y:y+h, x:x+w]
                    video[i] = frame
                else:
                    break
                i += 1
            videos.append(video)
        return videos

    def _concat_output(self, outp):
        return np.concatenate(outp, axis=0)

    def _visualize(self):
        frame = self.first_frame.copy()
        if self.crop is not None:
            x, y, w, h = self.crop[0], self.crop[1], self.crop[2], self.crop[3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=5)
        plt.imshow(frame, cmap='gray')
        plt.show()