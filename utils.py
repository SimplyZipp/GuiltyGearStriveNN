import time
import numpy as np
import cv2
from mss import mss
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SimpleTimer:
    """
    Simple, blocking timer
    """
    def __init__(self, interval, verbose=False):
        self.interval = interval
        self.time = 0
        self.verbose = verbose

    def start(self):
        self.time = time.time()

    def wait(self):
        remaining = self.interval - (time.time()-self.time)
        if remaining > 0:
            time.sleep(remaining)
        if self.verbose:
            print(f'Given interval is {self.interval}, time taken was {time.time()-self.time}')

    def wait_and_continue(self):
        self.wait()
        self.start()


class FrameGrabber:
    def __init__(self, size=(800, 450), bounds=None):
        if bounds is None:
            bounds = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.size = size
        self.bounds = bounds
        self.sct = mss()

        # Maybe normalize the image somehow?
        self.transform = A.Compose(
            [A.resize(self.size, interpolation=cv2.INTER_AREA),
             ToTensorV2()]
        )

    def frame(self):
        """
        Get a frame from the screen, resizing it to 'size' and making it grayscale
        :return: the modified frame image as a tensor
        """

        sct_img = self.sct.grab(self.bounds)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.transform(image=img)['image']
        return img
