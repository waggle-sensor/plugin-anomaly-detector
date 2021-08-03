#!/bin/env python3 
from waggle import plugin
from waggle.data.vision import Camera

import argparse
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf

from attention_cae import Attention_CAE
from online_models import OnlineAnomalyDetectionModel
from calibration_data import AnomalyDetectorBatcher

def main():
       
    parser = argparse.ArgumentParser(description="This program uses a convolutional autoencoder for online anomaly detection in images.")
    parser.add_argument("--interval", type=float, default=10.0, 
        help="time between inferences in seconds")
    parser.add_argument("--backup", type=int, default=40, 
        help="number of inferences between weight backups")
    parser.add_argument("--data", default="calibration_data", 
        help="directory containing calibration images")
    parser.add_argument("--weights", default="saved_weights", 
        help="directory containing saved weights (will ignore calibration data if present)")
    parser.add_argument("--buffsize", type=int, default=20, 
        help="number of images buffered in main memory. Making this number lower will increase performance but decrease accuracy.")
    parser.add_argument("--alpha", type=float, default=1e-4, 
        help="The estimated probability of an anomaly. Decreasing this number means fewer images will be flagged as anomalies")
    
    # initialize logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")
    logging.info("tensorflow version %s", tf.__version__)
    logging.info("opencv version %s", cv2.__version__)
    
    # initialize plugin:
    args = parser.parse_args() 
    plugin.init()
    camera = Camera()
    
    # use 4:3 input aspect ratio for input (height,width,channels):
    input_shape=(480,640,3)
    cae = Attention_CAE(input_shape=input_shape)
    model = OnlineAnomalyDetectionModel(
                cae,
                optimizer=tf.keras.optimizers.Adam(1e-5),
                input_shape=input_shape,
                alpha=1e-4, 
                example_buffer_len=20)
    
    # attempt to load saved weights (if not, calibrate model):
    logging.info('Looking for saved weights...')
    weights_path = os.path.join(args.weights, "model_weights.h5")
    gamma_filter_path = os.path.join(args.weights, "gamma_filter_weights.npy")
    try:
        model.load_weights(weights_path)
        model.load_gamma_filter(gamma_filter_path)
    except:
        logging.info('Unable to load saved weights. Re-calibrating model...')
        
        # calibrate model:
        calibration_data = AnomalyDetectorBatcher(
            normal_img_dir=os.path.join(args.data, 'normal'),
            anomaly_img_dir=os.path.join(args.data, 'anomaly'),
            image_shape=input_shape)
        
        if len(calibration_data) <= 0:
            
            logging.warning('WARNING: no calibration data was found.')
        else:
            model.calibrate(calibration_data, n_epochs=20)
            model.save_weights(weights_path)
            model.save_gamma_filter(gamma_filter_path)
        
    # get frames from camera stream (main loop):
    next_publish = time.time()
    publish_interval = args.interval
    for sample in camera.stream():
        now = time.time()
        if now >= next_publish:
            frame = sample.data
            s_x = frame.shape[1] / input_shape[1]
            s_y = frame.shape[0] / input_shape[0]
            resized_frame = cv2.resize(frame, (input_shape[1],input_shape[0]))
            resized_frame = resized_frame.astype(np.float32) / 255. 
            pred_boxes = model.consume_input(np.expand_dims(resized_frame,axis=0))
            resized_boxes = [ (s_x*x,s_y*y,s_x*w,s_y*h) for x,y,w,h in pred_boxes ]
            print(resized_boxes) 
            # publish predicted boxes:
            plugin.publish('anomaly-detector', resized_boxes)
            if resized_boxes:
                logging.info(f'Detected Anomalies at: {resized_boxes}')
            
            # set next publish time:
            next_publish = now + publish_interval

if __name__ == '__main__':
    main()
