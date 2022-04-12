import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'Data/haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('Data/face2.jpg')

# Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),
                  randrange(128, 256), randrange(128, 256)), 2)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

# Display the image with the faces spotted ('The app name',the image)
cv2.imshow('Face Detector', img)

# This will wait otherwise it will display the image and close and
# we will not be able to view it Properly and to close it press any Key.
cv2.waitKey()

print("Code Completed")
