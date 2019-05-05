#
from bson.binary import Binary
from imutils import paths
import numpy as np
import cv2
import os
import pickle
import pprint
from pymongo import MongoClient


def connect(host, port):
    """
    :param host: String, DB host port i.e. 'mongodb://localhost'
    :param port: Ing, Server port 'localhost default-27017
    :return: connection
    """
    return MongoClient(host, port)


def insert_user(user_id, recognize, motion):
    """
    :param user_id: Int, Users random generated ID
    :param recognize: Array, embeddings used to recognize user
    :param motion: Array, embeddings to be returned if user is recognized
    :return: Dictionary
    """
    post = {'user_id': user_id,
            'recognize': Binary(pickle.dumps(recognize, protocol=2)),
            'motion': Binary(pickle.dumps(motion, protocol=2))}
    return post


def retrieve_embeddings(user_id):
    """
    Retrieve embeddings from pickle file
    :param user_id: user_id you wish to push to server
    :return:
    """
    rec_emb = pickle.loads(open('./output/' + str(user_id) + '_rec_embeddings.pickle', 'rb').read())
    motion_emb = pickle.loads(open('./output/' + str(user_id) + '_mot_embeddings.pickle', 'rb').read())
    return rec_emb, motion_emb


def test_load(user_id):
    """
    Get user info from database
    Unpickle Binary to array
    :param user_id: Int, user_id in server DB you wish to find and test
    :return: None
    """
    u = users.find_one({'user_id': int(user_id)})
    k = pickle.loads(u['recognize'])
    print(k)


if __name__ == '__main__':
    client = connect('mongodb://localhost', 27017)
    db = client.SeniorDesign
    users = db.users
    id = input('User_id: ')
    rec_embeddings, motion_embeddings = retrieve_embeddings(id)

    put = insert_user(int(id), rec_embeddings, motion_embeddings)
    user_id = users.insert(put)

    # todo create batch push function
    # %timeit -n 100 [pickle.loads(x['user_id']) for x in collection.find()]
    # test_load(10140) # Uncomment to test response from server enter user ID
