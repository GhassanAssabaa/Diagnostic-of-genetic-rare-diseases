import pickle
import json
import numpy 
import time

from sklearn import svm
from sklearn import metrics
from joblib import load

from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector


def init():
    global model

    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name = 'Trisomy_21_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    
def run(raw_data):
    data = np.array(raw_data)
	Rechaped_data = A.reshape(1, -1)
	result = loaded_model.predict(B)