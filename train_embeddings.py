# todo no longer need this file. Not doing scv classification
# CSE Senior Design 2019
# Face Classification Training
# Only used if implementing SCV

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

data = pickle.loads(open('./output/embeddings.pickle', 'rb').read())

# Sklearn label encoder for classification
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['id'])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
recognizer = SVC(C=1.0, kernel="linear", probability=True)

recognizer.fit(data['embeddings'], labels)

# Save face SVM.SVC recognition model
f = open('./output/recognizer.pickle', 'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open('./output/le.pickle', 'wb')
f.write(pickle.dumps(label_encoder))
f.close()
