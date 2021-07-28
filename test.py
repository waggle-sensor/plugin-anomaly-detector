import tensorflow as tf

from unittest import main, TestCase
from calibration_data import AnomalyDetectorBatcher
from online_models import OnlineAnomalyDetectionModel
from attention_cae import Attention_CAE

class DatasetCalibrationTest(TestCase):

    def test_load_dataset(self):
        pass

    def test_data_calibration(self):
        input_shape = (960,1280,3) 
        model = OnlineAnomalyDetectionModel(
                lossmap_model=Attention_CAE(),
                optimizer=tf.keras.optimizers.Adam(1e-5),
                input_shape=input_shape)
         
        adb = AnomalyDetectorBatcher(
                normal_img_dir='test/normal',
                anomaly_img_dir='test/anomaly',
                image_shape=input_shape)
        
         

if __name__ == '__main__':
    main()
