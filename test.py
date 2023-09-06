from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

video = cv2.VideoCapture(1)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

data_directory = r'C:\Users\shaha\OneDrive\Desktop\Notes\Projects\face_recognition_project\data'
with open(os.path.join(data_directory, 'names.pkl'), 'rb') as w:
    LABELS = pickle.load(w)
with open(os.path.join(data_directory, 'faces_data.pkl'), 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

attendance_directory = r'C:\Users\shaha\OneDrive\Desktop\Notes\Projects\face_recognition_project\Attendance'

attendance = []  # List to store attendance data

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    
    if key == ord('o') and len(faces) > 0:
        speak("Attendance Taken..")
        time.sleep(5)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        for (x, y, w, h) in faces:
            attendance.append([str(output[0]), str(timestamp)])  # Store the recognized face and timestamp
        attendance_flag = True
    
    if key == ord('q'):
        break

if attendance:
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    csv_file_path = os.path.join(attendance_directory, 'Attendance_' + date + '.csv')
    if os.path.isfile(csv_file_path):
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for data in attendance:
                writer.writerow(data)
    else:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
            for data in attendance:
                writer.writerow(data)
    print(f'Attendance saved to {csv_file_path}')

video.release()
cv2.destroyAllWindows()


