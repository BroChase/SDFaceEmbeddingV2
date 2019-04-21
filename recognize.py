import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # comment out to run on gpu
import cv2
import numpy as np
import pickle
import dlib
from scipy.spatial import distance
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.video import VideoStream
import time
from keras.models import load_model
import utils
from itertools import chain
from create_embeddings import encode_stream

# load the dlib face detector.
detector = dlib.get_frontal_face_detector()

# Load the saved model.
model = load_model('face-rec_Google.h5')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)


def recognize_face(face_descriptor, database):
    encoding = encode_stream(face_descriptor, model)

    db_enc = list(database.values())
    temp = 0.1
    identity = None
    dist = None
    # Loop over the database dictionary's names and encodings.
    for i in range(len(db_enc[0])):
        # todo look up linear subtraction from rows to calculate the minimum embedding distance in one go 'no loop'
        dist = np.linalg.norm(db_enc[0][i] - encoding)
        # todo test different distance thresh holds
        # todo !! calculate distance between C and N Retrain C
        if dist < 0.1:
            print(dist)
            if dist < temp:
                temp = dist
                identity = db_enc[1][i]
            #return identity, dist
        # else:
        #     return None, 0
    if temp != None:
        return identity, dist
    else:
        return None, 0


def recognize():
    # load database
    database = pickle.loads(open('./output/embeddings.pickle', 'rb').read())

    # Start camera with warmup timer
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        # Capture the stream and convert to gray scale. Try to detect a face.
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        # If face is detected
        x = 0
        y = 0
        w = 0
        h = 0
        if len(faces) >= 1:
            # todo Crashing when detecting more than one face, Filter out other faces. Refer to old code.
            face = faces[0]
            # If more than one face is detected get the largets of the set.
            for i in range(len(faces)):
                (_x, _y, _w, _h) = face_utils.rect_to_bb(faces[i])
                if _w > w or _h > h:
                    face = faces[i]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            # Align the detected face using imutils face alignment.
            face_img = face_aligner.align(img, img_gray, face)
            name, min_dist = recognize_face(face_img, database)
            if min_dist < 0.08:
                cv2.putText(img, "Face : " + str(name), (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Frame", img)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # destroy windows and stop stream
    cv2.destroyAllWindows()
    cap.release()


recognize()
