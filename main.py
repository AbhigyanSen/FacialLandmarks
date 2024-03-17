import dlib
import cv2
import numpy as np

# Model path (replace with the path where you downloaded the model)
Model_PATH = "shape_predictor_68_face_landmarks.dat"

# Load the face detector and landmark detector
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

def draw_landmarks(frame, landmarks):
  """Draws facial landmarks on the frame.

  Args:
    frame: The frame to draw on.
    landmarks: The detected facial landmarks.
  """
  for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  if not ret:
    print("Error: Unable to capture frame")
    break

  # Convert frame to RGB format (dlib expects RGB)
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
    draw_landmarks(frame, detectedLandmarks)

  # Display the resulting frame
  cv2.imshow('Facial Landmarks', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) == ord('q'):
    break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()