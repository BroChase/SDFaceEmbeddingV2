# Create_embeddings of users for database. Not needed for recognize.py todo implement on server side
# Searches over the dataset creating embeddings for each image of each user in the dataset.
# Stores the resulting dictionary in a pickle file for easy load/dump

import os
import pickle
import numpy as np
import cv2
from imutils import paths
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment out to run on GPU.
from keras.models import load_model

# State Saved Face Recognition model.
FRmodel = load_model('FaceRecognition.h5')


def embeddings():
    user_embeddings = []
    user_identifier = []
    # todo rework for server
    imagePaths = list(paths.list_images('./dataset'))
    # Enumerate over all of the images in the Dataset creating embeddings for each user.
    for (i, imagePath) in enumerate(imagePaths):
        user_id = imagePath.split(os.path.sep)[-2]
        # append users id
        user_identifier.append(user_id)
        # append the embeddings for that user_id
        user_embeddings.append(encode_image(imagePath, FRmodel).flatten())
    # Store the embeddings for use.
    database = {"embeddings": user_embeddings, "id": user_identifier}
    # save the embeddings to a pickle file
    f = open('./output/embeddings.pickle', "wb")
    f.write(pickle.dumps(database))
    f.close()


# Creates the actual encoding. 128-d vector.
def encode_image(image_path, model):
    # load the image and resize it.
    img = cv2.imread(image_path, 1)
    image = cv2.resize(img, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    image_data = np.array([img])
    # pass the image into the model and predict. 'forward pass' Returns 128-d vector.
    embedding = model.predict(image_data)
    return embedding


embeddings()
