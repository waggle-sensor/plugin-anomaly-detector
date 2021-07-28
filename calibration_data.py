import tensorflow as tf
from tensorflow.keras.utils import Sequence

from glob import glob
import os
import random

class AnomalyDetectorBatcher(Sequence):

    def __init__(self, normal_img_dir, anomaly_img_dir, image_shape, shuffle=True):

        assert(image_shape[0] > 0)
        assert(image_shape[1] > 0)

        EXTENSIONS = ['*.png', '*.jpeg', '*.jpg']

        self.data = []
        self.image_shape = image_shape

        for ext in EXTENSIONS:
            glb = glob(os.path.join(normal_img_dir, ext))
            for path in glb:
                self.data.append((path, False))

        for ext in EXTENSIONS:
            glb = glob(os.path.join(anomaly_img_dir, ext))
            for pth in glb:
                self.data.append((path,True))


        # optinally shuffle data:
        if shuffle:
            random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, y = self.data[idx]
        im_x = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        im_x = np.array(cv2.resize(im_x, (self.image_shape[0],self.image_shape[1]))) / 255.
        im_x = np.expand_dims(im_x, axis=0).astype(np.float32)

        return im_x, y

