import cv2
import numpy as np
import tensorflow as tf

# Load classifier
model = tf.keras.models.load_model("models/cat_dog_model.h5")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to the required input size for our classifier
    resized_frame = cv2.resize(gray, (256, 256))
    # Convert the resized frame to RGB
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
    # Get the predictions from the classifier
    predictions = model.predict(np.expand_dims(resized_frame/255, 0))

    # Put the prediction on the frame
    if predictions < 0.5:
        cv2.putText(frame, "Cat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "Dog", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Live Camera Feed", frame)
    # Break the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
