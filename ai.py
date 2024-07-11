from keras.api.models import load_model
import numpy
import cv2

def model():
    return load_model(r'C:\programming\sharing\firedetect\submodel.keras')

def predict(img : numpy.ndarray, models) -> bool:
    #fire -> True, non fire -> False

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = numpy.float32(img) / 255.0
    img = numpy.expand_dims(img, axis=0)
    
    result = models.predict(img, batch_size = 32)
    
    return result[0][0] < 0.5