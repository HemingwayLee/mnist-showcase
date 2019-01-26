import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier 

# The digits dataset 8x8
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# Flatten the image, turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 2 layer, first layer 16 units, second layer 12 units
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 12), random_state=1)

# Training: the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Testing: the second half of the digits:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
