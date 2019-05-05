# Facial recognition --  Senior Design 2019
# Waits for interrupt call from greeter to begin authentication attempts.
# When face is detected in stream.
# --Sends embeddings to DB server to search and find user if located in system.
# --Upon return of a user being recognized attempts to match motion embedding in real time.
# If motion embedding threshold met
# --Returns greeter -Authenticated -User ID

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # comment out to run on gpu
from bson.binary import Binary
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from keras.models import load_model
import numpy as np
import pickle
import scipy.spatial.distance as distance
import socket
import time

# load the dlib face detector.
detector = dlib.get_frontal_face_detector()

# Load the saved model. (No model.? RUN 'train_model.py')
model = load_model('FaceRecognition.h5')

# Shape predictor for facial alignment using landmarks.
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Aligns Face by feature extraction from shape_predictor
face_aligner = FaceAligner(shape_predictor)


# todo move function to server side for recognition
def recognize_face(face_descriptor, database):
    # Calculate norm between users in database and incoming user verification embedding.

    # create embeddings of image.
    encoding = encode_stream(face_descriptor, model)
    # Database of encodings todo Store embeddings created in DB
    db_enc = list(database.values())
    temp = 0.1
    identity = None
    dist = None
    # Loop over the database dictionary's ID and encodings.
    for i in range(len(db_enc[0])):

        if db_enc[1][i] == '14565_user':
            print(dist)
        # todo play with thresh .002-.003ish 'need user data'
        if dist < 0.003:
            if dist < temp:
                temp = dist
                identity = db_enc[1][i]

    if identity is not None:
        return identity, dist
    else:
        return None, 0


# Used in recognize.py to encode the stream images.
def encode_stream(img, model):
    # load the image and resize it.
    image = cv2.resize(img, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    image_data = np.array([img])
    # pass the image into the model and predict. 'forward pass' Returns 128-d vector.
    embedding = model.predict(image_data)
    return embedding


# **Facenet distance metrics function https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
# todo test diference in cosine similarity vrs sum of the square difference
def distance_metric(embeddings1, embeddings2, metric=0):
    """
    :param embeddings1: database embedding
    :param embeddings2: real time user embedding
    :param metric: 0,1
    :return: distance 'similarity'
    """
    if metric==0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif metric==1: # todo fix axis rotations
        dist = distance.cosine(embeddings1, embeddings2)
    else:
        raise Exception('Undefined metric')
    return dist


def client():
    host = socket.gethostname()  # get local machine name
    port = 8080  # Make sure it's within the > 1024 $$ <65535 range

    s = socket.socket()
    s.connect((host, port))

    message = input('-> ')
    while message != 'q':
        s.send(message.encode('utf-8'))
        data = s.recv(1024).decode('utf-8')
        print('Received from server: ' + data)
        message = input('==> ')
    s.close()


def recognize():
    # Loops camera feed input checking for a user
    # If a face or faces are detected in feed attempt to verify
    # todo **
    # load database todo put on database end for user verification. Embeddings will be stored in Database.
    # database = pickle.loads(open('./embeddings.pickle', 'rb').read())
    # todo **
    # Start camera with warm up timer of 2'sec'

    host = socket.gethostname()
    port = 8080
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))



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

            # send embedding to server for authentication
            s.send(data_string)


            # ****************************
            data_arr = b""
            while True:
                data = s.recv(1024)
                if not data:
                    break
                data_arr += data
            emb = pickle.loads(data_arr)
            print('test')
            s.close()
            # *****************************
            # # todo **
            # # Call recognize_face function to create embedding and compare norm vs user db
            # # todo Rework. Create embeddings. Send embeddings to DB for comparison. Return User ID and motion Embedding.
            # # todo need to think about batching for more accurate detection. N out of 10 captures images == USER return
            # name, min_dist = recognize_face(face_img, database)
            # # todo **
            # if min_dist < 0.08:
            #     cv2.putText(img, "Face : " + str(name), (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            #     cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            # else:
            #     cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        # Show cam feed
        # cv2.imshow("Frame", img)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # Clean up--destroy windows and stop stream
    cv2.destroyAllWindows()
    cap.release()
    s.close()


if __name__ == '__main__':
    recognize()
