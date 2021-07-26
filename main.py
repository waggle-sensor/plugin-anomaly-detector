from waggle import plugin
from waggle.data.vision import Camera

import cv2
import tensorflow as tf

from attention_cae import AttentionCAE
from online_models import OnlineAnomalyDetectionModel

def main():
    plugin.init()
    camera = Camera()
    
    # use 4:3 input aspect ratio:
    input_shape=(960,1280,3)
    cae = AttentionCAE(input_shape=input_shape)
    
    model = OnlineAnomalyDetectionModel(
                cae,
                optimizer=tf.keras.optimizers.Adam(1e-5),
                input_shape=input_shape,
                alpha=1e-4,
                batch_size=4,
                example_buffer_len=20)
    
    # calibrate model:
    

    for frame in camera.stream()
        frame = cv2.resize(frame, (input_shape[1],input_shape[0]))
        
