# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')
# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Create an linear SVM object
clf = LinearSVC()

# Splitting the training and testing set
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.1, random_state=42)

# Perform the training
clf.fit(X_train, y_train)

# Perform evaluation
print('The prediction error is' + '{:.1%}'.format(1-clf.score(X_test,y_test)))