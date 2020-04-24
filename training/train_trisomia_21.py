import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess

import sklearn
from sklearn import svm
from sklearn import metrics
from joblib import dump
from typing import Tuple, List


from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
datafile = get_absPath("Final.csv")
# check that file exists
data = pd.read_csv(datafile)
data = data[['chin_0_x','chin_0_y','chin_1_x','chin_1_y','chin_2_x','chin_2_y','chin_3_x','chin_3_y','chin_4_x','chin_4_y','chin_5_x','chin_5_y','chin_6_x','chin_6_y','chin_7_x','chin_7_y','chin_8_x','chin_8_y','chin_9_x','chin_9_y','chin_10_x','chin_10_y','chin_11_x','chin_11_y','chin_12_x','chin_12_y','chin_13_x','chin_13_y','chin_14_x','chin_14_y','chin_15_x','chin_15_y','chin_16_x','chin_16_y','left_eyebrow_0_x','left_eyebrow_0_y','left_eyebrow_1_x','left_eyebrow_1_y','left_eyebrow_2_x','left_eyebrow_2_y','left_eyebrow_3_x','left_eyebrow_3_y','left_eyebrow_4_x','left_eyebrow_4_y','right_eyebrow_0_x','right_eyebrow_0_y','right_eyebrow_1_x','right_eyebrow_1_y','right_eyebrow_2_x','right_eyebrow_2_y','right_eyebrow_3_x','right_eyebrow_3_y','right_eyebrow_4_x','right_eyebrow_4_y','nose_bridge_0_x','nose_bridge_0_y','nose_bridge_1_x','nose_bridge_1_y','nose_bridge_2_x','nose_bridge_2_y','nose_bridge_3_x','nose_bridge_3_y','nose_tip_0_x','nose_tip_0_y','nose_tip_1_x','nose_tip_1_y','nose_tip_2_x','nose_tip_2_y','nose_tip_3_x','nose_tip_3_y','nose_tip_4_x','nose_tip_4_y','left_eye_0_x','left_eye_0_y','left_eye_1_x','left_eye_1_y','left_eye_2_x','left_eye_2_y','left_eye_3_x','left_eye_3_y','left_eye_4_x','left_eye_4_y','left_eye_5_x','left_eye_5_y','right_eye_0_x','right_eye_0_y','right_eye_1_x','right_eye_1_y','right_eye_2_x','right_eye_2_y','right_eye_3_x','right_eye_3_y','right_eye_4_x','right_eye_4_y','right_eye_5_x','right_eye_5_y','top_lip_0_x','top_lip_0_y','top_lip_1_x','top_lip_1_y','top_lip_2_x','top_lip_2_y','top_lip_3_x','top_lip_3_y','top_lip_4_x','top_lip_4_y','top_lip_5_x','top_lip_5_y','top_lip_6_x','top_lip_6_y','top_lip_7_x','top_lip_7_y','top_lip_8_x','top_lip_8_y','top_lip_9_x','top_lip_9_y','top_lip_10_x','top_lip_10_y','top_lip_11_x','top_lip_11_y','bottom_lip_0_x','bottom_lip_0_y','bottom_lip_1_x','bottom_lip_1_y','bottom_lip_2_x','bottom_lip_2_y','bottom_lip_3_x','bottom_lip_3_y','bottom_lip_4_x','bottom_lip_4_y','bottom_lip_5_x','bottom_lip_5_y','bottom_lip_6_x','bottom_lip_6_y','bottom_lip_7_x','bottom_lip_7_y','bottom_lip_8_x','bottom_lip_8_y','bottom_lip_9_x','bottom_lip_9_y','bottom_lip_10_x','bottom_lip_10_y','bottom_lip_11_x','bottom_lip_11_y','Trisomy 21']]

predict = 'Trisomy 21'

print("Columns:", data.columns) 
print("Final data set dimensions : {}".format(data.shape))


X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print("Training the model...")


clf = svm.SVC(kernel="linear") #On peut donné en paramètre kernel and soft margin of the hyperplan
clf.fit(x_train, y_train)



print("Evaluate the model...")
y_pred = clf.predict(x_test) # Predict values for our test data
acc = metrics.accuracy_score(y_test, y_pred) # Test them against our correct values

print("The test accuracy of the model is :", acc)
for x in range(len(y_pred)):
    print('The predicted value is ',y_pred[x],'    ','The value in the datasat is :', y_test[x])

# Save model as part of the run history
print("Exporting the model as sav or pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "Trisomy_21_model.sav"
model_path = os.path.join(outputs_folder, model_filename)
pickle.dump(clf, open(model_path, 'wb'))


# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()