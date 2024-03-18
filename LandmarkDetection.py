# Facial Landmark Detection on Images

import dlib
import cv2
import numpy as np
from facePoints import facePoints
import os

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
  with open(fileName, 'w') as f:
    for p in faceLandmarks.parts():
      f.write("%s %s\n" %(int(p.x),int(p.y)))
  f.close()

Model_PATH = "shape_predictor_68_face_landmarks.dat"

# Face Detector and Landmark Detector
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# FileName Functionality
def store_FileName(FilePath):
    filename_with_extension = os.path.basename(FilePath)
    FileName, _ = os.path.splitext(filename_with_extension)
    return FileName

FilePath = "Images\Input\DemoImage1.png"          # Change Accordingly
FileName = store_FileName(FilePath)
print("Filename without extension:", FileName)

img = cv2.imread(FilePath)
imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Storing LandMarks
faceLandmarksOuput= f"Images/Output/{FileName}"  # Change Accordingly

# Face Detection
allFaces = frontalFaceDetector(imageRGB, 0)
print("List of all faces detected: ",len(allFaces))

# List to store Landmarks of all Detected Faces
allFacesLandmark = []

for k in range(0, len(allFaces)):
  faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
      int(allFaces[k].right()),int(allFaces[k].bottom()))

  detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
  
  # Number of Landmarks Detected on Image
  if k==0:
    print("Total number of face landmarks detected ",len(detectedLandmarks.parts()))

  # Saving Landmarks
  allFacesLandmark.append(detectedLandmarks)

  # Drawing Landmarks
  facePoints(img, detectedLandmarks)

  fileName = faceLandmarksOuput +"_"+ str(k)+ ".txt"
  print("Lanmdark is save into ", fileName)

  writeFaceLandmarksToLocalFile(detectedLandmarks, fileName)

# Saving Output Image with Detected Landmarks
outputNameofImage = f"Images/Output/{FileName}.jpg"
print("Saving output image to", outputNameofImage)
cv2.imwrite(outputNameofImage, img)

cv2.imshow("Face landmark result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
