import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response

# Load the trained emotion recognition model
model_best = load_model(r'C:\Users\USER\Desktop\Projects\Emotion recognizer\Emotion_Recognizer\models/face_model.h5')

# Classes (7 emotional states)
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Flask app
app = Flask(__name__)

# Initialize the video capture object (only once)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    raise Exception("Could not open video device")

# Function to capture video frames
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if frame capture fails

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]

            # Resize the face image to the required input size for the model
            face_image = cv2.resize(face_roi, (48, 48))  # Ensure this matches your model input size
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.vstack([face_image])  # Stack to create a batch

            # Predict emotion using the loaded model
            predictions = model_best.predict(face_image)
            emotion_label = class_names[np.argmax(predictions)]

            # Display the emotion label on the frame
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Encode the frame in JPEG format and yield it to be sent to the frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the video feed (emotion recognition)
@app.route('/emotion_recognition')
def emotion_recognition():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()  # Release the capture device when the app stops
