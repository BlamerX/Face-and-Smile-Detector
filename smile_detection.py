import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
face_detector = cv2.CascadeClassifier(
    'Data/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('Data/haarcascade_smile.xml')

# To Detect faces from webcam.
webcam = cv2.VideoCapture(0)

# Iterate forever over frame
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # If there's an error, abort
    if not successful_frame_read:
        break

    # Convert to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Run the face detector within each of these faces
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the sub frame (using numpy N-diminsional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detects smiles
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)  # scaleFactors,min_neighbours

        # # Find all the smiles in the face
        # for (x_,y_,w_,h_) in smiles:
        #     # Draw a rectangle around the smile
        #     cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)

        # Label this face as Smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smilling', (x, y+h+20), fontScale=1,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Show the current frame
    cv2.imshow('Smile Detector', frame)

    # This will wait otherwise it will display the image and close and
    # we will not be able to view it Properly and to close it press any Key.
    key = cv2.waitKey(1)  # 1ms

    # Stop if Q Key is pressed
    if key == 81 or key == 113:
        break

# Release VideoCapture Object
webcam.release()

# To destroy all windows
cv2.destroyAllWindows()


print("Code Completed")
