import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('b.jpg')

# Capture Video from webcam
webcam = cv2.VideoCapture(0)

while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw Rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 5)

    # Display the image with the faces
    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)

    # Stop is 0 key is pressed
    if key == 81 or key == 113:  # q OR Q
        break

# Release the VideoCapture object
webcam.release()

print("\n Code Completed\n")
