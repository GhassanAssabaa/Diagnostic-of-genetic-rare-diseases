from flask import *
from sklearn import svm
from sklearn import metrics
import pickle
import pandas as pd
import numpy as np
import face_recognition
import cv2

def input_visage(List_visage):
    A = np.array(List_visage)
    B = A.reshape(1, -1)
    filename = 'finalized_model2.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(B)
    return(result)

def extract_features(Filename):
    # Collecte des informations d'une seule image

    image = face_recognition.load_image_file(Filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    L = face_recognition.face_landmarks(image)
    for i in range(0, len(L)):
        # Séparation des features en plusieurs listes
        chin = L[0]["chin"]
        left_eyebrow = L[0]["left_eyebrow"]
        right_eyebrow = L[0]["right_eyebrow"]
        nose_bridge = L[0]["nose_bridge"]
        nose_tip = L[0]["nose_tip"]
        left_eye = L[0]["left_eye"]
        right_eye = L[0]["right_eye"]
        top_lip = L[0]["top_lip"]
        bottom_lip = L[0]["bottom_lip"]
        # Création de la liste final qui va être stocké dans le fichier
        List_visage = []
        for i in range(0, len(chin)):
            for j in range(0, len(chin[i])):
                List_visage.append(chin[i][j])

        for i in range(0, len(left_eyebrow)):
            for j in range(0, len(left_eyebrow[i])):
                List_visage.append(left_eyebrow[i][j])

        for i in range(0, len(right_eyebrow)):
            for j in range(0, len(right_eyebrow[i])):
                List_visage.append(right_eyebrow[i][j])

        for i in range(0, len(nose_bridge)):
            for j in range(0, len(nose_bridge[i])):
                List_visage.append(nose_bridge[i][j])

        for i in range(0, len(nose_tip)):
            for j in range(0, len(nose_tip[i])):
                List_visage.append(nose_tip[i][j])

        for i in range(0, len(left_eye)):
            for j in range(0, len(left_eye[i])):
                List_visage.append(left_eye[i][j])

        for i in range(0, len(right_eye)):
            for j in range(0, len(right_eye[i])):
                List_visage.append(right_eye[i][j])

        for i in range(0, len(top_lip)):
            for j in range(0, len(top_lip[i])):
                List_visage.append(top_lip[i][j])

        for i in range(0, len(bottom_lip)):
            for j in range(0, len(bottom_lip[i])):
                List_visage.append(bottom_lip[i][j])

    return List_visage