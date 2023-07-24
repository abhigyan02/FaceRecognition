from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

olivetti_data = fetch_olivetti_faces()

# there are 400 images - 10x40 (40 people - 1 person has 10 images) - 1 image = 64x64 pixels
features = olivetti_data.data
# we represent target variables (people) with integers (face ids)
targets = olivetti_data.target

fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
sub_plots = sub_plots.flatten()

for unique_user_id in np.unique(targets):
    image_index = unique_user_id * 8
    sub_plots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
    sub_plots[unique_user_id].set_xticks([])
    sub_plots[unique_user_id].set_yticks([])
    sub_plots[unique_user_id].set_title('Face id: %s' % unique_user_id)

plt.suptitle('The dataset (40 people)')

# let's plot the 10 images for the first person (face id =0)
fig, sub_plot = plt.subplots(nrows=1, ncols=10, figsize=(14, 8))

for j in range(10):
    sub_plot[j].imshow(features[j].reshape(64, 64), cmap='gray')
    sub_plot[j].set_xticks([])
    sub_plot[j].set_yticks([])
    sub_plot[j].set_title('Face id=0')

plt.show()

# split the original data-set (training and test set)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

# let's try to find optimal number of eigenvectors (principle components)
pca = PCA(n_components=100, whiten=True)
pca.fit(X_train)
X_pca = pca.fit_transform(features)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

models = [('Logistic Regression', LogisticRegression()), ('Support Vector Machine', SVC()), ("Naive Bayes Classifier", GaussianNB())]

for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_score = cross_val_score(model, X_pca, targets, cv=kfold)
    print(name, "- Mean of the cross-validation score: %s" % cv_score.mean())
