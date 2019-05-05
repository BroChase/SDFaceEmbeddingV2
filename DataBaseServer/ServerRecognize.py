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
from threading import Thread


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

        for i in range(len(k)):
            dist = distance_metric(k[i], embedding, 1)
            if dist < 0.005:
                if dist < temp:
                    temp = dist
                    user_id = ident
                    user_motion = m

    if user_id is not None:
        return user_motion  #user_id, user_motion, dist
    else:
        return 'nouser'


# **Facenet distance metrics function https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
# todo test diference in cosine similarity vrs sum of the square difference
def distance_metric(embeddings1, embeddings2, metric=0):
    """
    :param embeddings1: database embedding
    :param embeddings2: real time user embedding
    :param metric: 0,1
    :return: distance 'similarity'
    """
    if metric == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif metric == 1:
        dist = distance.cosine(embeddings1, embeddings2)
    else:
        raise Exception('Undefined metric')
    return dist


class Server:
    def __init__(self, host, port):
        # connection to mongodb
        self.client = connect('mongodb://localhost', 27017)
        self.db = self.client.SeniorDesign
        self.users = self.db.users
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)

    def listen_for_clients(self):
        while True:
            client, addr = self.server.accept()
            print('Connection from: ' + str(addr[0]) + ':' + str(addr[1]))
            Thread(target=self.handle_client, args=(client, addr)).start()

    def handle_client(self, client_socket, address):
        while True:
            try:
                # data_array = b""
                data = client_socket.recv(4096)
                print('test')
                emb = pickle.loads(data)

                if 'q^' in emb:
                    break
                else:
                    # emb = pickle.loads(data)
                    user_motion = recognize_face(emb, self.users)
                    data_string = pickle.dumps(user_motion)
                    client_socket.send(data_string)
                    break

            except socket.error:
                client_socket.close()
                return False

        client_socket.close()


def server():
    """
    Establish connection to mongoDB
    Listen for requests from client
    Respond back with motion payload if Authenticated.
    :return: None
    """

    host = socket.gethostname()
    port = 8080
    main = Server(host, port)
    # start listening for clients
    main.listen_for_clients()


if __name__ == '__main__':
    server()
