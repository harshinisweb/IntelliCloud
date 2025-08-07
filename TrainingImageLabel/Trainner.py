import cv2
import numpy as np
import os
from PIL import Image

# Path where images are stored
path = "TrainingImage"

# Create LBPH Face Recognizer
recognizer =cv2.face.LBPHFaceRecognizer_create()

# Haarcascade classifier for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get images and labels for training
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert("L")  # Convert to grayscale
        img_numpy = np.array(PIL_img, "uint8")

        # Extract student ID from file name
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Detect face in the image
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, np.array(ids)

# Get training data
faces, ids = getImagesAndLabels(path)

# Train the recognizer
recognizer.train(faces, ids)

# Save trained model as 'Trainner.yml'
recognizer.save("TrainingImageLabel/Trainner.yml")

print("Training completed successfully! Trainner.yml file created.")
