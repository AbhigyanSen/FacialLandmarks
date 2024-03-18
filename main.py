import dlib
import cv2
import numpy as np

Model_PATH = "shape_predictor_68_face_landmarks.dat"

# Loading the Face Detector and Landmark Detector
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# WebCam Capture
cap = cv2.VideoCapture(0)

def draw_landmarks(frame, landmarks):
  for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

while True:
  # Capturing frame-by-frame
  ret, frame = cap.read()

  if not ret:
    print("Error: Unable to capture frame")
    break
    
  frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Detect Faces in the Frame
  allFaces = frontalFaceDetector(frameRGB, 0)

  # Process Detected Frame
  for k in range(0, len(allFaces)):
    faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                       int(allFaces[k].right()), int(allFaces[k].bottom()))

    # Facial Landmarks
    detectedLandmarks = faceLandmarkDetector(frameRGB, faceRectangleDlib)
    draw_landmarks(frame, detectedLandmarks)

  # Display Image with Landmarks
  cv2.imshow('Facial Landmarks', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
