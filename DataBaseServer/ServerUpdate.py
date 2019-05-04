#

from imutils import paths
import numpy as np
import cv2
import os
import pickle
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
            'rec': recognize,
            'motion': motion}
    return post


if __name__ == '__main__':
    client = connect('mongodb://localhost', 27017)
    db = client.SeniorDesign
    users = db.users