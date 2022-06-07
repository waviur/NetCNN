import cv2
from keras.models import Sequential
from keras.saving.save import load_model

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = Sequential()
model = load_model('net.test.h5')

model.summary()

image = cv2.imread("horse.jpeg")
output = image.copy()
image = cv2.resize(image, (32, 32))

image = image.astype('float32') / 255.0
image = image.reshape((1, 32, 32, 3))

result = model.predict(image)

label_index = result.argmax(axis=1)[0]
label = labels[label_index]

text = "{}: {:.2f}%".format(label, result[0][label_index] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Image", output)
cv2.waitKey(0)
