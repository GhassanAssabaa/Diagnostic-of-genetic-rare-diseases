from flask import *
from sklearn import svm
from sklearn import metrics
import pickle
import pandas as pd
import numpy as np
import face_recognition
import cv2
import functions
import os
from os.path import join as pjoin
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory


app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    global List
    target2 = os.path.join(APP_ROOT, 'images/',filename)
    List = functions.extract_features(target2)
    Output = functions.input_visage(List)
    if (Output == 1):
        result = 'Trisomy 21 founded'
    elif (Output == 2):
        result = 'Angel man Syndrome founded'
    elif (Output == 3):
        result = 'Williams Syndrome founded'
    else:
        result = 'No Syndrome founded'
    if (Output == 1):
        description = 'Down syndrome, also known as trisomy 21, is a genetic disorder caused by the presence of all or part of a third copy of chromosome 21. It is usually associated with physical growth delays, mild to moderate intellectual disability, and characteristic facial features'
        frec = 'Down syndrome occurs in about 1 in 800 newborns. About 5,300 babies with Down syndrome are born in the United States each year, and approximately 200,000 people in this country have the condition. Although women of any age can have a child with Down syndrome, the chance of having a child with this condition increases as a woman gets older.'
    elif (Output == 2):
        description = 'Angelman syndrome is a complex genetic disorder that primarily affects the nervous system. Characteristic features of this condition include delayed development, intellectual disability, severe speech impairment, and problems with movement and balance (ataxia). Most affected children also have recurrent seizures (epilepsy) and a small head size (microcephaly). Delayed development becomes noticeable by the age of 6 to 12 months, and other common signs and symptoms usually appear in early childhood.'
        frec = 'Angelman syndrome affects an estimated 1 in 12,000 to 20,000 people.'
    elif (Output == 3):
        description = 'Williams syndrome is a developmental disorder that affects many parts of the body. This condition is characterized by mild to moderate intellectual disability or learning problems, unique personality characteristics, distinctive facial features, and heart and blood vessel (cardiovascular) problems. \n People with Williams syndrome typically have difficulty with visual-spatial tasks such as drawing and assembling puzzles, but they tend to do well on tasks that involve spoken language, music, and learning by repetition (rote memorization). Affected individuals have outgoing, engaging personalities and tend to take an extreme interest in other people. Attention deficit disorder (ADD), problems with anxiety, and phobias are common among people with this disorder.'
        frec = 'Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.'
    else:
        description = 'No description '
        frec = 'Normal person'
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("final.html", name=result, desc = description, desc2 = frec, image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
