import tensorflow as tf

from unittest import main, TestCase
from calibration_data import AnomalyDetectorBatcher
from online_models import OnlineAnomalyDetectionModel
from attention_cae import Attention_CAE

class CalibrationTest(TestCase):

    NORMAL_IMG_DIR = 'test/normal'
    ANOMALY_IMG_DIR = 'test/anomaly'
    N_IMAGES = 8
    INPUT_SHAPE = (960,1280,3) 

    def test_load_dataset(self):
        data = AnomalyDetectorBatcher(
                    CalibrationTest.NORMAL_IMG_DIR,
                    CalibrationTest.ANOMALY_IMG_DIR,
                    image_shape=CalibrationTest.INPUT_SHAPE)
         
        # ensure proper length:
        assert(len(data) == CalibrationTest.N_IMAGES)
        
        # ensure all images are resized in batches of 1:
        for img, label in data:
            assert(img.shape == ((1,) + CalibrationTest.INPUT_SHAPE))

    def test_data_calibration(self):
        data = AnomalyDetectorBatcher(
                    CalibrationTest.NORMAL_IMG_DIR,
                    CalibrationTest.ANOMALY_IMG_DIR,
                    image_shape=CalibrationTest.INPUT_SHAPE)
        
        model = OnlineAnomalyDetectionModel(
                lossmap_model=Attention_CAE(),
                optimizer=tf.keras.optimizers.Adam(1e-5),
                input_shape=CalibrationTest.INPUT_SHAPE)
        
        # perform a single epoch of training: 
        model.calibrate(data, n_epochs=1)

        

if __name__ == '__main__':
    main()
