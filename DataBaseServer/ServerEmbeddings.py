from imutils import paths
import numpy as np
import cv2
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment out to run on GPU.
from keras.models import load_model


# Creates the actual encoding. 128-d vector.
def encode_image(image_path, model):
    """
    :param image_path: OS.PATH, location of images
    :param model: MODEL, model for forward pass, creates embeddings
    :return: Array, 128d vector of floats -- embeddings from image.
    """
    # load the image and resize it.
    img = cv2.imread(image_path, 1)
    image = cv2.resize(img, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    image_data = np.array([img])
    # pass the image into the model and predict. 'forward pass' Returns 128-d vector.
    embedding = model.predict(image_data)
    return embedding


def de_pickle(data, motion, batch=0):
    """
    Create embeddings and store in pickle files.-- ref. create_embeddings.
    :param data: OS.PATH, location of user_id meta data
    :param motion: OS.PATH, location of user_id motion meta data
    :param batch: Int, 1 Batch process all data in ./database and ./motiondata, else only user_id files
    :return: Arrays, recognition embeddigs and movement embeddings
    """
    FRmodel = load_model('../FaceRecognition.h5')

    if batch == 0:
        user_id = input('Enter user_id: ')
        user_recognition = []
        user_motion = []
        # user_identifier = []
        recognition_images = list(paths.list_images(data + '/' + str(user_id) + '/'))
        motion_images = list(paths.list_images(motion + '/' + str(user_id) + '/'))
        # Create embeddings pickle from recognition meta data
        for (i, imagePath) in enumerate(recognition_images):
            # todo add if statment to check input id is os path id
            user_id = imagePath.split(os.path.sep)[-2]
            # append users id
            # user_identifier.append(user_id)
            # append the embeddings for that user_id
            user_recognition.append(encode_image(imagePath, FRmodel).flatten())
        for (i, imagePath) in enumerate(motion_images):
            user_id = imagePath.split(os.path.sep)[-2]
            # append user id
            # user_identifier.append(user_id)
            # append the embeddings for user_id
            user_motion.append(encode_image(imagePath, FRmodel).flatten())

        # Store the embeddings for use.
        database = {'id': user_id, 'recognition': user_recognition, 'motion': user_motion}
        # save the embeddings to a pickle file
        f = open('./output/' + str(user_id) + '_embeddings.pickle', 'wb')
        f.write(pickle.dumps(database))
        f.close()
    elif batch == 1:
        user_embeddings = []
        user_identifier = []
        recognition_images = list(paths.list_images(data))
        motion_images = list(paths.list_images(motion))
        # Enumerate over all of the images in the Dataset creating embeddings for each user.
        # Create embeddings pickle from recognition meta data
        for (i, imagePath) in enumerate(recognition_images):
            user_id = imagePath.split(os.path.sep)[-2]
            # append users id
            user_identifier.append(user_id)
            # append the embeddings for that user_id
            user_embeddings.append(encode_image(imagePath, FRmodel).flatten())
        # Store the embeddings for use.
        database = {'embeddings': user_embeddings, 'id': user_identifier}
        # save the embeddings to a pickle file
        f = open('./output/rec_embeddings.pickle', 'wb')
        f.write(pickle.dumps(database))
        f.close()
        # clear lists
        user_embeddings = []
        user_identifier = []
        # Create embeddings pickle from motion meta data
        for (i, imagePath) in enumerate(motion_images):
            user_id = imagePath.split(os.path.sep)[-2]
            # append user id
            user_identifier.append(user_id)
            # append the embeddings for user_id
            user_embeddings.append(encode_image(imagePath, FRmodel).flatten())
        # Store the embeddings
        database = {'embeddings': user_embeddings, 'id': user_identifier}
        # save the embeddings to a pickle file
        f = open('./output/mot_embeddings.pickle', 'wb')
        f.write(pickle.dumps(database))
        f.close()


if __name__ == '__main__':
    DATA_DIR = './dataset'
    MOTI_DIR = './motiondata'
    # 0 for single 1 for Batch
    de_pickle(DATA_DIR, MOTI_DIR, 0)
