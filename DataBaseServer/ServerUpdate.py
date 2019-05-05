# Senior Design Spring 2019
# Creates connection to DB
# Prompts admin for user_id
# Finds the pickle file of the user located in DataBaseServer/output and loads the data.
# Inserts user to DB
from bson.binary import Binary
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
            'recognize': Binary(pickle.dumps(recognize, protocol=2)),
            'motion': Binary(pickle.dumps(motion, protocol=2))}
    return post


def retrieve_embeddings(user_id):
    """
    Retrieve embeddings from pickle file
    :param user_id: user_id you wish to push to server
    :return: Arrays, recognition array of arrays, motion array of arrays
    """
    emb = pickle.loads(open('./output/' + str(user_id) + '_embeddings.pickle', 'rb').read())
    rec = emb['recognition']
    mot = emb['motion']
    return rec, mot


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
