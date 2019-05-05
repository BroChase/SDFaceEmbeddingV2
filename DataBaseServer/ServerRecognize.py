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
import socket


def recognize_face(embedding, users):
    temp = 0.1
    user_id = None
    user_motion = None
    dist = None
    # Loop over the database dictionary's ID and encodings.
    for u in users.find():
        ident = u['user_id']
        k = pickle.loads(u['recognize'])
        m = pickle.loads(u['motion'])
        db_enc = list(k.values())

        for i in range(len(db_enc[0])):
            dist = distance_metric(db_enc[0][i], embedding, 1)
            if dist < 0.003:
                if dist < temp:
                    temp = dist
                    user_id = ident
                    user_motion = m

    if user_id is not None:
        return user_motion #user_id, user_motion, dist
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


def server():
    # connection to mongodb
    client = connect('mongodb://localhost', 27017)
    db = client.SeniorDesign
    users = db.users
    # connection to client greeter
    host = socket.gethostname()
    port = 8080
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    conn, addr = s.accept()
    while True:
        data = conn.recv(4096)
        emb = pickle.loads(data)
        break
        if not data:
            break
    # user_id, user_motion, dist = recognize_face(emb, users)
    user_motion = recognize_face(emb, users)
    # user_motion = np.array([[1,3,4,5,65,], [3,7,6,5,4,5,4]])
    data_string = pickle.dumps(user_motion)
    conn.send(data_string)
    conn.close()


    # listen for traffic
    # s.listen(1)
    # client_socket, address = s.accept()
    # print("Connection from: " + str(address))
    # while True:
    #     data = s.recv(8192)
    #     emb = pickle.loads(data)
    #     #user_id, user_motion, dist = recognize_face(emb, users)
    #     #data = [user_id, user_motion, dist]
    #     b = pickle.dumps(emb)
    #     if not data:
    #         break
    #     s.send(b)
    #
    # s.close()


if __name__ == '__main__':
    server()
