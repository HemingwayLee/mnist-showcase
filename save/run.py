import cv2
from tensorflow import keras
import matplotlib.pyplot as plt

(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_test.shape)
print(y_test.shape)
print(x_test[:1])
print(y_test[:10])

for i in range(8):
    print(y_test[i])
    print(x_test[i].shape)
    cv2.imwrite(f"{y_test[i]}.png", x_test[i])
    plt.subplot(2, 4, i+1)
    plt.imshow(x_test[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis("off")

plt.show()

