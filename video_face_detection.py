import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'Data/haarcascade_frontalface_default.xml')

# To Detect faces from webcam.
webcam = cv2.VideoCapture(0)

# Iterate forever over frame
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with the faces spotted ('The app name',the image)
    cv2.imshow('Face Detector', frame)

    # This will wait otherwise it will display the image and close and
    # we will not be able to view it Properly and to close it press any Key.
    key = cv2.waitKey(1)  # 1ms

    # Stop if Q Key is pressed
    if key == 81 or key == 113:
        break

# Release VideoCapture Object
webcam.release()

print("Code Completed")
