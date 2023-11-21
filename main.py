import cv2
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import time

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Largeur de la vidéo
cap.set(4, 480)  # Hauteur de la vidéo
cv2.namedWindow("python webcam")
# Mesurer le temps de capture
while True:
    ret, frame = cap.read()

    # Prétraitement de l'image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = np.reshape(gray, (1, 28, 28, 1))
    gray = gray / 255.0  # Normalisation

    # Faire une prédiction
    prediction = model.predict(gray)
    predicted_label = np.argmax(prediction)

    # Afficher le chiffre prédit
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-time Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Appuyez sur la touche 'Esc' pour quitter
        break

cap.release()
cv2.destroyAllWindows()