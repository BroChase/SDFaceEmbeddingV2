# Create meta data of users
# dataset - meta data for recognition
# motiondata - meta data for motion recognition
#
# to use:
# Run Create_user.py
# Randomly generates user ID
# First Prompt: Capture 20 images for user recognition "NO MOVEMENT"
# --Saved to dataset under users random generated ID
# Second Prompt: Capture 60 images for user motion recognition "MOVEMENTS"
# --Saved to motiondata under users random generated ID
# **** Email user ID to participant for if they ever want their embeddings removed from the dataset***

# Requires shape_predictor_68_face_landmarks.dat file

import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import os
from random import randint
import time

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# face alignment from imutils.
face_aligner = FaceAligner(shape_predictor)

# default data directory for training.
DATA_DIR = './dataset/'
MOTION_DIR = './motiondata/'


def still_images(cap, user_folder):
    """
    Captures still images for user recognition
    :param cap: VideoCapture
    :param user_folder: OS.PATH, Location of user recognition data
    :return: None
    """
    total_imgs = 20
    image_number = 0
    print('Now Capturing User')
    while True:
        # Capture the stream and convert to gray scale. Try to detect a face.
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        h = 0
        w = 0
        # If face is detected
        if len(faces) >= 1:
            # Set face to first
            face = faces[0]
            # If more than one face is detected select largest in set.
            for i in range(len(faces)):
                (_x, _y, _w, _h) = face_utils.rect_to_bb(faces[i])
                if _w > w or _h > h:
                    face = faces[i]
            face_img = face_aligner.align(img, img_gray, face)
            # Set the path variable to save image.
            img_path = user_folder + str(image_number) + ".jpg"
            cv2.imwrite(img_path, face_img)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            # Need to warm up the camera in order for it so show the captured image.
            image_number += 1
            print(image_number)

        cv2.waitKey(1)
        if image_number == total_imgs:
            print('Capture Complete')
            break


def motion_images(cap, motion_folder):
    """
    Capture users motion images
    :param cap: VideaoCapture
    :param motion_folder: OS.PATH, Location of motion data
    :return: None
    """
    total_imgs = 60
    image_number = 0
    print('Now Capturing Motion')
    while True:
        # Capture the stream and convert to gray scale. Try to detect a face.
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        h = 0
        w = 0
        # If face is detected
        if len(faces) >= 1:
            # Set face to first
            face = faces[0]
            # If more than one face is detected select largest in set.
            for i in range(len(faces)):
                (_x, _y, _w, _h) = face_utils.rect_to_bb(faces[i])
                if _w > w or _h > h:
                    face = faces[i]
            face_img = face_aligner.align(img, img_gray, face)
            # Set the path variable to save image.
            img_path = motion_folder + str(image_number) + ".jpg"
            cv2.imwrite(img_path, face_img)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            # Need to warm up the camera in order for it so show the captured image.
            image_number += 1
            print(image_number)

        cv2.waitKey(1)
        if image_number == total_imgs:
            print('Capture Complete')
            break


def create():
    """
    Capture frames of Video stream.
    Randomly generates user_id int-range of 10000-20000
    Prompt user if they are ready to capture still frames. If yes, capture '20 frames of user being still'
        --Saved to dataset folder
    Prompt user if they are ready to capture motion frames. If yes, capture '60 frames of user doing motion'
        --Saved to motiondata folder
    :return:
    """
    # Start Stream
    cap = cv2.VideoCapture(0)
    # Warm up camera
    time.sleep(2.0)
    # Capture and create user metadata.
    # user = 'user'
    # motion = 'motion'
    user_id = randint(10000, 20000)
    print(f'User ID: {user_id}')
    user_folder = None
    motion_folder = None
    # Check if default dataset directory exists else create
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(MOTION_DIR):
        os.mkdir(MOTION_DIR)
    # # Create a folder for training for user
    while True:
        try:
            user_folder = DATA_DIR + str(user_id) + '/'
            motion_folder = MOTION_DIR + str(user_id) + '/'
            if not os.path.exists(user_folder):
                os.mkdir(user_folder)
            if not os.path.exists(motion_folder):
                os.mkdir(motion_folder)
            break
        except:
            print('Error creating directory')
            continue
    # Recognition images
    ready = input('Ready to capture(Y/N)')
    if ready == 'Y' or ready == 'y':
        still_images(cap, user_folder)
    # Motion images
    ready = input('Ready to capture(Y/N)')
    if ready == 'Y' or ready == 'y':
        motion_images(cap, motion_folder)
    # release the videoCapture
    cap.release()


if __name__ == '__main__':
    create()
