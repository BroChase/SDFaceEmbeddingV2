# Senior Design Spring 2019
# Waits for interrupt call from greeter to begin authentication attempts.
# When face is detected in stream.
# --Sends embeddings to DB server to search and find user if located in system.
# --Upon return of a user being recognized attempts to match motion embedding in real time.
# If motion embedding threshold met
# --Returns greeter -Authenticated -User ID
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # comment out to run on gpu
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from keras.models import load_model
import numpy as np
import pickle
import scipy.spatial.distance as distance
import socket
import sys
import time

# load the dlib face detector.
detector = dlib.get_frontal_face_detector()

# Load the saved model. (No model.? RUN 'train_model.py')
model = load_model('FaceRecognition.h5')

# Shape predictor for facial alignment using landmarks.
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Aligns Face by feature extraction from shape_predictor
face_aligner = FaceAligner(shape_predictor)


def encode_stream(img, model):
    """
    encode the image using the saved model.
    :param img: Jpg, Image of users face
    :param model: Frozen DNN
    :return: Array, embeddings
    """
    # load the image and resize it.
    image = cv2.resize(img, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    image_data = np.array([img])
    # pass the image into the model and predict. 'forward pass' Returns 128-d vector.
    embedding = model.predict(image_data)
    return embedding


def client(data):
    """
    Sends embedding to server for authentication
    :param data: Pickle, 'array',
    :return: Array, Int, motion embeddings and user_id
    """
    target_host = socket.gethostname()
    target_port = 8080
    try:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Could not create a socket')
        time.sleep(1)
        sys.exit()

    try:
        c.connect((target_host, target_port))
    except socket.error:
        print('Could not connect to server')
        time.sleep(1)
        sys.exit()

    Online = True
    while Online:
        c.send(data)
        data_arr = b""
        while True:
            data = c.recv(1024)
            if not data:
                break
            data_arr += data
        emb = pickle.loads(data_arr)
        Online = False
        c.close()
    return emb  # todo fix reference problem


def recognize():
    """
    Loops camera feed input checking for a user
    If a face or faces are detected in feed attempt to verify
    :return: None
    """
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    # Loop till user is recognized in feed.
    while True:
        # Capture the stream and convert to gray scale. Try to detect a face.
        ret, img = cap.read()
        # Color image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use
        faces = detector(img_gray)
        # If face is detected
        w = 0
        h = 0
        if len(faces) >= 1:
            # Set face to first
            face = faces[0]
            # If more than one face is detected select largest in set.
            for i in range(len(faces)):
                (_x, _y, _w, _h) = face_utils.rect_to_bb(faces[i])
                if _w > w or _h > h:
                    face = faces[i]
            # Get bounding box of the detected face.
            (x, y, w, h) = face_utils.rect_to_bb(face)
            # Align the detected face using face_aligner
            face_img = face_aligner.align(img, img_gray, face)
            encoding = encode_stream(face_img, model)
            data_string = pickle.dumps(encoding)

            emb = client(data_string)  # todo test more.
            # todo create and compare motion matrices
            """
            If emb != q^ -- then skip sending the above line to client()
            Begin to store frames in queue like structure. 
            When queue fills [60] frames
            Compare with emb
            Continue to collect frames pushing out the old frames and updating it with the new frame to keep at [60].
            Continue to check for similarity between queue and motion embeddings.
            If under threshold 
            return to greeter -- AUTH and ID for user login.
            """

            # Uncomment for visual window
            # if min_dist < 0.08:
            #     cv2.putText(img, "Face : " + str(name), (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            #     cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            # else:
            #     cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            # Show Cam feed in window
            # cv2.imshow("Frame", img)

        key = cv2.waitKey(1) & 0xFF
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # Clean up--destroy windows and stop stream
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    recognize()
