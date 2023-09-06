import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(1)

# Define the absolute path to the Haar Cascade XML file
cascade_path = r'C:\Users\shaha\OneDrive\Desktop\Notes\Projects\face_recognition_project\data\haarcascade_frontalface_default.xml'

# Check if the file exists before creating the CascadeClassifier
if not os.path.isfile(cascade_path):
    print(f"Error: The file '{cascade_path}' does not exist.")
    exit()

facedetect = cv2.CascadeClassifier(cascade_path)

data_directory = r'C:\Users\shaha\OneDrive\Desktop\Notes\Projects\face_recognition_project\data'

faces_data = []
names = []

i = 0

name=input("Enter Your Name: ")

# Add the name to the 'names' list only once
names.extend([name] * 100)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Save the collected data and labels
with open(os.path.join(data_directory, 'names.pkl'), 'wb') as f:
    pickle.dump(names, f)

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

with open(os.path.join(data_directory, 'faces_data.pkl'), 'wb') as f:
    pickle.dump(faces_data, f)
