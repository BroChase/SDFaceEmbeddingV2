from bson.binary import Binary
from DataBaseServer.ServerUpdate import connect
from imutils import paths
import numpy as np
import cv2
import os
import pickle
import pprint
from pymongo import MongoClient
import scipy.spatial.distance as distance


def recognize_face(embedding, users):
    temp = 0.1
    identity = None
    dist = None
    # Loop over the database dictionary's ID and encodings.
    for u in users.find():
        k = pickle.loads(u['recognize'])
        dist = distance_metric(k, embedding, 1)
        if dist < 0.003:
            if dist < temp:
                temp = dist
                identicy = u['user_id']

    if identity is not None:
        return identity, dist
    else:
        return None, 0


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


if __name__ == '__main__':
    client = connect('mongodb://localhost', 27017)
    db = client.SeniorDesign
    users = db.users

    