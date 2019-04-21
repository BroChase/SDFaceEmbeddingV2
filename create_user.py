import cv2
import os
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

detector = dlib.get_frontal_face_detector()
# todo look at creating higher face landmarks.dat file 'http://dlib.net/train_shape_predictor.py.html'
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# face alignment from imutils. todo test other widths.
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)

# default data directory for training.
DATA_DIR = 'dataset/'


def create():
    # Check if default dataset directory exists else create
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    # Get Users name and assign them an ID number.
    while True:
        user_name = input('User Name: ')
        try:
            user_id = int(input('User ID: '))
            break
        except:
            print(f'Invalid user id {user_id} ')
            continue
    # Create a folder for training for user
    while True:
        try:
            user_folder = DATA_DIR + str(user_id) + '_' + user_name + "/"
            if not os.path.exists(user_folder):
                os.mkdir(user_folder)
            break
        except:
            print('Error creating directory')
            continue

    image_number = 0
    # Start Stream
    cap = cv2.VideoCapture(0)
    # Number of images to save from stream.
    total_imgs = 1
    # todo 'Stagger the image capture so they are not all from a 1 second snap'
    while True:
        # Capture the stream and convert to gray scale. Try to detect a face.
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        # If face is detected
        if len(faces) == 1:
            # todo Filter out other faces. Refer to old code.
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            # Align the detected face using imutils face alignment.
            face_img = face_aligner.align(img, img_gray, face)
            # Set the path variable to save image.
            img_path = user_folder + str(image_number) + ".jpg"
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            # Need to warm up the camera in order for it so show the captured image.
            image_number += 1
        cv2.waitKey(1)
        if image_number == total_imgs:
            break
    # release the stream.
    cap.release()

if __name__ == '__main__':
    create()
