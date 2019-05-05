# Senior Design Spring 2019
# 3D plot at least 1 point from each user in the database

# todo. Connect to database and get recognition embeddings and plot on 3D plane for demonstration of spatial distance
# todo between users

import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


database = pickle.loads(open('./output/embeddings.pickle', 'rb').read())

ak = list(database.values())

x = ak[0]

pca = PCA(n_components=3)
pca.fit(x)
q = pca.transform(x)
qc = q[:10]
qm = q[10:19]
qn = q[19:]

fig = plt.figure()
ax = plt.axes(projection='3d')
# todo better iteration function for plotting colors based on user
for i in range(len(q)):
    zdata = q[i][0]
    xdata = q[i][1]
    ydata = q[i][2]
    ax.scatter3D(xdata, ydata, zdata, c='r')


plt.show()
