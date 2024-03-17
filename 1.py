import dlib
import cv2
import numpy as np
from facePoints import facePoints  # Assuming facePoints function draws landmarks

# Model Path (download the model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
Model_PATH = "shape_predictor_68_face_landmarks.dat"

# Load the face detector and landmark detector
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# Start video capture from webcam (replace 0 with video file path if desired)
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  if not ret:
    print("Error: Unable to capture frame")
    break

  # Convert frame to RGB format
  frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Detect faces in the frame
  allFaces = frontalFaceDetector(frameRGB, 0)

  # Process each detected face
  for k in range(0, len(allFaces)):
    # Define face rectangle
    faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                       int(allFaces[k].right()), int(allFaces[k].bottom()))

    # Detect facial landmarks
    detectedLandmarks = faceLandmarkDetector(frameRGB, faceRectangleDlib)

    # Draw landmarks on the frame
    facePoints(frame, detectedLandmarks)

  # Display the resulting frame
  cv2.imshow('Facial Landmarks', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) == ord('q'):
    break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()