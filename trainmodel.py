import cv2
import os
import numpy as np

def train_model():
    # Initialize LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Path to the folder containing images for training
    image_folder = 'TrainingImage/'

    # Initialize arrays to hold image data (faces) and labels (student IDs)
    faces = []
    labels = []

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop through each image file in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Only process image files (e.g., .jpg, .png)
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            print(f"Processing image: {image_name}")
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            # If no faces are detected, skip this image
            if len(faces_detected) == 0:
                print(f"No faces detected in image: {image_name}")
                continue

            # Loop through each detected face
            for (x, y, w, h) in faces_detected:
                face = gray[y:y + h, x:x + w]
                faces.append(face)

                # Extract student ID from the filename (e.g., 'studentname.serialno.id.jpg')
                # Example: 'JohnDoe.123.456.jpg' => serialno = 123
                filename_parts = image_name.split('.')
                if len(filename_parts) < 3:  # Ensure there are at least 3 parts (name.serialno.id)
                    print(f"Skipping invalid filename: {image_name}")
                    continue  # Skip invalid filenames

                try:
                    # The serialno should be in the second part of the filename
                    label = int(filename_parts[1])  # Extracting serialno from the filename
                except ValueError:
                    print(f"Invalid serialno in filename: {image_name}")
                    continue  # Skip invalid files where serialno can't be converted to int

                labels.append(label)

    # Check if enough faces were collected for training
    if len(faces) == 0:
        print("No faces found for training. Ensure you have images with faces.")
        return

    # Train the LBPH recognizer with the faces and labels
    recognizer.train(faces, np.array(labels))

    # Save the trained model to a file
    recognizer.save('TrainingImageLabel/Trainner.yml')

    print("Model trained and saved successfully!")

# Call the function to start training the model
train_model()
