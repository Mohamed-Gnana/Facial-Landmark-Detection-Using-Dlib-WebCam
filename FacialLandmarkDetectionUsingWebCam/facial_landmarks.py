from myUtils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# Fetching the predictor and the image
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")

args = vars(ap.parse_args())

# dlib Detector and Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Capturing images using WebCam
video = cv2.VideoCapture(0)


while True:
    ret, frame = video.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the faces
    facesAsRectangles = detector(grayFrame, 1)
    # loop over the face detections
    for (i, rect) in enumerate(facesAsRectangles):
        shape = predictor(grayFrame, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_opencvbox(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), 2)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Video", frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
video.release()
cv2.destroyAllWindows()

