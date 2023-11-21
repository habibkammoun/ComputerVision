import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("modelCNN.h5")

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width of the video
cap.set(4, 480)  # Height of the video

while True:
    ret, frame = cap.read()

    # Preprocess the image
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to match the input size of the model
    img_array = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display the predicted class
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-time Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
