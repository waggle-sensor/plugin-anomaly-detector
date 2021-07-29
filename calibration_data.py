import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
import numpy as np

from glob import glob
import os
import random

class AnomalyDetectorBatcher(Sequence):
    """
        This is a batcher that loads a dataset dynamically
        (i.e. without loading all images into memory at once)

        The batch size of the batcher is fixed at 1 to reduce memory
        usage.
    """
    
    
    def __init__(self, normal_img_dir, anomaly_img_dir, image_shape, shuffle=True):
        """
            Constructs a AnomalyDatasetBatcher.

            parameters:
                normal_img_dir: a directory containing "normal" images

                anomaly_img_dir: a directory containing "anomaly" images

                image_shape: the shape of the input tensor: (height,width,n_channels)
                              (recommended is (960,1280,3))
                
                shuffle: whether or not to shuffle the data
        """
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
        
        
        # optionally shuffle data:
        if shuffle:
            random.shuffle(self.data)
        
        
    def __len__(self):
        """ returns length of dataset """
        return len(self.data)

    def __getitem__(self, idx):
        """ returns the item at idx """
        path, y = self.data[idx]
        im_x = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        im_x = np.array(cv2.resize(im_x, (self.image_shape[1],self.image_shape[0]))) / 255.
        im_x = np.expand_dims(im_x, axis=0).astype(np.float32)
        
        return im_x, y

