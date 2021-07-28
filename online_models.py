import numpy as np
import tensorflow as tf
import cv2

import sklearn
import scipy.stats as stats
import scipy.special as sps
from tqdm.notebook import tqdm

import os
import time
from glob import glob
import random
from parse import parse
from collections import deque
import datetime
import pickle as pkl
from itertools import islice

from samplers import ExponentialHeapSampler

class OnlineAnomalyDetectionModel():
    """
        Represents an Online Anomaly Detection Model.

        This class wraps a tensorflow model that computes
        an 'anomaly heatmap' given an image. This heatmap
        could be an autoencoder model, where the output
        values correspond to the per-pixel reconstruction
        loss.
    """    

    def __init__(self, lossmap_model, optimizer, input_shape, 
                 alpha=0.005, beta=0.005, 
                 example_buffer_len=20):
        """
            Constructs an Online Anomaly Detection model.

            parameters:
                
                lossmap_model: a lossmap model (see notes below).
                
                optimizer: default optimizer for online training
                           (recommended: tf.keras.optimizers.Adam(1e-5) )
                
                input_shape: the input image tensor shape, with channels
                             (e.g. (960,1280,3))
                
                alpha: the estimated probability of an anomaly
                       (this is like the 'alpha' value in a p-value test)
                
                beta: decay coefficient for gamma distribution filter
                      (recommended: 0.005)
                              
                example_buffer_len: the number of buffered example images
                                    (recommended: 20)
            
            Note: the lossmap_model must conform to the followng interface:
            
            # train_step(self, x, optimizer) -> model_loss
        
            # predict(self, x)  -> reconstruction_error_lossmap
            
            # save_weights(self, path)
            
            # load_weights(self, path)
            
            (It is recommended this be an instance of tf.keras.models.Model)
        """
        assert(0.0 <= alpha <= 1.0)
        assert(0.0 <= beta <= 1.0)
        assert(example_buffer_len > 0)
        assert(input_shape[0] > 0)
        assert(input_shape[1] > 0)
        assert(input_shape[2] > 0)
        
        self.lossmap_model = lossmap_model
        self.input_sampler = ExponentialHeapSampler(example_buffer_len)
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta
        self.gamma_kappa = None
        self.gamma_theta = None
        
        self.gamma_x = None
        self.gamma_logx = None
        self.gamma_xlogx = None
    
    def calibrate(self, data, n_epochs=40, optimizer=None, verbose=True):
        """
            Calibrates an online model (to avoid the model thinking everything is
            an anomaly right after deployment).
    
            parameters:
                
                data: an array of (batched) input tensors [could be a tf.keras.utils.Sequence]
                
                n_epochs: number of training epochs
        
                optimizer: optimizer for training (defaults to model's optimizer)
        
                verbose: train with verbose output
        """

        if optimizer == None:
            optimizer = self.optimizer
        
        for i in range(n_epochs):
            start_time = time.time()
            loss = tf.keras.metrics.Mean()
            for x,y in tqdm(data):
                loss(self.lossmap_model.train_step(x, optimizer))
            train_loss = loss.result()
            end_time = time.time()
            time_delta = end_time - start_time
            if verbose:
                print(f'Epoch {i+1} Train loss: {train_loss:.8f} ({time_delta:.2f} s)')
        
        gamma_sum_x = None
        gamma_sum_logx = None
        gamma_sum_xlogx = None
        
        if verbose:
            print('Computing gamma filter parameters ...')
        for x,y in tqdm(data):
            lossmap = self.lossmap_model.predict(x)
            lossmap = np.clip(lossmap, 1e-30, None)
            log_lossmap = np.log(lossmap)
            if gamma_sum_x is None:
                gamma_sum_x = lossmap
                gamma_sum_logx = log_lossmap
                gamma_sum_xlogx = lossmap*log_lossmap
            else:
                gamma_sum_x += lossmap
                gamma_sum_logx += log_lossmap
                gamma_sum_xlogx += lossmap*log_lossmap
        
        n_x = len(data)
        self.gamma_theta = (n_x*gamma_sum_xlogx - gamma_sum_x*gamma_sum_logx) / (n_x*n_x)
        kappa_denom = np.clip((n_x*gamma_sum_xlogx - gamma_sum_x*gamma_sum_logx), 1e-30, 1e30)
        self.gamma_kappa = (n_x*gamma_sum_x) / kappa_denom
        
        self.gamma_x = gamma_sum_x / n_x
        self.gamma_logx = gamma_sum_logx / n_x
        self.gamma_xlogx = gamma_sum_xlogx / n_x
        
        if verbose:
            print('Computing confusion matrix ...')
        confusion_mat = {'TP' : 0, 'FP' : 0, 'TN' : 0, 'FN' : 0}
        for x, y in tqdm(data):
            boxes = self._identify_lossmap_boxes(lossmap)
            if boxes and y:
                confusion_mat['TP'] += 1
            elif boxes and not y:
                confusion_mat['FP'] += 1
            elif (not boxes) and y:
                confusion_mat['FN'] += 1
            else:
                confusion_mat['TN'] += 1
            
            # prime the input sampler with calibration images:
            if not y:
                self.input_sampler.add(x)
            
        if verbose:
            mean_kappa = np.mean(self.gamma_kappa)
            mean_theta = np.mean(self.gamma_theta)
            print(f'\nConfusion Matrix (alpha = {self.alpha}): ')
            for k, n in confusion_mat.items():
                print(f'{k} Count: {n}')
            print(f'Gamma filter means:\ntheta: {mean_theta}\nkappa: {mean_kappa}')
            
        return confusion_mat
        
            
    def consume_input(self, x, optimizer=None):
        """
            processes a single new frame, which triggers a single training step on
            all buffered examples in the model's example buffer

            parameters:
                x: the new frame as an image tensor
                
                optimizer: the optimizer to use (defaults to model's optimizer)
        """
        lossmap = self.lossmap_model.predict(x)
        pred_boxes = self._identify_lossmap_boxes(lossmap)
        
        if optimizer == None:
            optimizer = self.optimizer

        # record input as not-anomalous:
        self.input_sampler.add(x)

        # perform a weight update iteration:
        for x in self.input_sampler:
            self.lossmap_model.train_step(x, optimizer)
        
        # update gamma filter
        beta_c = (1.0 - self.beta)
        clip_lm = np.clip(lossmap, 1e-30, None)
        self.gamma_x = self.beta*clip_lm + beta_c*self.gamma_x
        self.gamma_logx = self.beta*np.log(clip_lm) + beta_c*self.gamma_logx
        self.gamma_xlogx = self.beta*(clip_lm*np.log(clip_lm)) + beta_c*self.gamma_xlogx
        
        self.gamma_theta = self.gamma_xlogx - self.gamma_logx*self.gamma_x
        self.gamma_kappa = self.gamma_x / self.gamma_theta
        
        #print(np.mean(self.gamma_kappa), np.mean(self.gamma_theta))
        
        return pred_boxes

    def save_gamma_filter(self, path):
        """
            Saves this model's gamma distribution filter to the given path
            (recommended to use ".npy" file extension)
        """
        with open(path, 'wb') as f:
            np.save(f, self.gamma_kappa)
            np.save(f, self.gamma_theta)
            np.save(f, self.gamma_x)
            np.save(f, self.gamma_logx)
            np.save(f, self.gamma_xlogx)
        
    def load_gamma_filter(self, path):
        """
            Loads this model's gamma distribution filter weights
        """
        with open(path, 'rb') as f:
            self.gamma_kappa = np.load(f)
            self.gamma_theta = np.load(f)
            self.gamma_x = np.load(f)
            self.gamma_logx = np.load(f)
            self.gamma_xlogx = np.load(f)
            
    def save_weights(self, path):
        """
            Saves the weights of this model to the given path
            (recommended to use ".h5" extension)
        """
        self.lossmap_model.save_weights(path)
        
    def load_weights(self, path):
        """
            Loads this model's weights
        """
        self.lossmap_model.load_weights(path)
    
    def predict(self, x):
        """
            Predicts the bounding boxes of any anomalies (if any) for a
            single image (without modifying any weights)
        
            parameters:
                x: input image tensor
        """
        lossmap = self.lossmap_model.predict(x)
        pred_boxes = self._identify_lossmap_boxes(lossmap)
        return pred_boxes    

    def _batch_groups(self, x_list):
        """
            helper function to return a batch iterator (unused)
        """
        x_it = iter(x_list)
        while True:
            chunk = tuple(islice(x_it, self.batch_size))
            if not chunk:
                return
            batch_chunk = np.concatenate(chunk, axis=0)
            yield batch_chunk
    
    def _identify_lossmap_boxes(self, lm, min_area_frac=0.0005, max_n_objs=5):
        """
            helper function to identify bounding boxes from a lossmap
        """
        assert(lm.shape[0] == 1)
        lossmap  = lm[0]
        im_size = lossmap.shape[0]*lossmap.shape[1]
        gamma_mass = np.squeeze(stats.gamma.cdf(lossmap, a=self.gamma_kappa, 
                                     scale=self.gamma_theta))
        _, thresh_lossmap = cv2.threshold(gamma_mass,(1.0-self.alpha),1.0,cv2.THRESH_BINARY)
        thresh_lossmap = cv2.erode(thresh_lossmap, cv2.getStructuringElement(0, (5,5)))
        thresh_lossmap = cv2.dilate(thresh_lossmap, cv2.getStructuringElement(0, (11,11)))
        thresh_lossmap = thresh_lossmap.astype(np.uint8)
        
        cnts, _ = cv2.findContours(thresh_lossmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = sorted([
            (cv2.boundingRect(c),cv2.contourArea(c))
            for c in cnts ],
            key= lambda x : -x[1])
        cnts = [ ca[0] for ca in cnt_list if ca[1] > min_area_frac*im_size ]
        return cnts[:min(max_n_objs, len(cnts))]
        
