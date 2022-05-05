import cv2
import numpy as np
import dlib

#detector
detector = dlib.get_frontal_face_detector()

#predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#image
img = cv2.imread("Xiong Naijin-19-crop-0.jpg")

# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    x = face.left() # left
    y = face.top() # top
    x2 = face.right() # right
    y2 = face.bottom() # bottom 

    marker = predictor(image=gray, box=face)

    for i in range(0, 68):
        x = marker.part(i).x
        y = marker.part(i).y

        cv2.circle(img=img, center=(x, y), radius=2, color=(255, 0, 0), thickness=-2)

cv2.imshow(winname="Face", mat=img)

cv2.waitKey(delay=0)

cv2.destroyAllWindows()