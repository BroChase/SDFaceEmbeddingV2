# Senior Design Spring 2019
# 3D plot at least 1 point from each user in the database


from DataBaseServer.ServerUpdate import connect
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib import animation
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import pickle
from random import sample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

client = connect('mongodb://localhost', 27017)
db = client.SeniorDesign
users = db.users


def NormConfMatrix():
    emb = []
    id = []
    for u in users.find():
        k = pickle.loads(u['recognize'])
        for i in k:
            emb.append(i)
            id.append(u['user_id'])

    x_df = pd.DataFrame(emb)
    y_df = pd.DataFrame(id)
    c = np.array([i for i in id])

    # df = pd.concat([x_df, y_df], axis=1, sort=False)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.20)


    neigh = KNeighborsClassifier()
    neigh.fit(x_train, y_train)

    y_pred = neigh.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    classes = unique_labels(c)
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    title = 'Normalized User Confusion Matrix'

    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True User ID', xlabel='Predicted User ID')
    plt.xticks(rotation='vertical')
    plt.show()
    # fig.savefig('NormConfMatrix')


if __name__ == '__main__':
    NormConfMatrix()
