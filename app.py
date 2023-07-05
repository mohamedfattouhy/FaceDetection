# MANAGE ENVIRONMENT
import os
import numpy as np
import cv2
from flask import Flask, render_template, Response
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__, template_folder='.')

model_path = os.path.join('facedetection', 'trained_model',
                          'facetracker.h5')
facetracker = load_model(model_path)


# Function to use the trained model on camera images
def face_detection_webcam():

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized/255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:

            # Controls the main rectangle
            cv2.rectangle(frame,
                        tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                        (255, 0, 0), 2)

            # Controls the label rectangle
            cv2.rectangle(frame,
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), 
                            [0, -30])),
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                            [80, 0])),
                        (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, 'Face',
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                        [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 1, cv2.LINE_AA)

            probability_detection = yhat[0][0][0]
            probability_detection = round(probability_detection, 3)

            # Controls the text rendered
            cv2.putText(frame, str(probability_detection),
                        tuple(np.add(np.multiply(sample_coords[2:], [450, 450]).astype(int),
                        [-130, 35])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')


@app.route('/')
def index():
    return render_template('static/index.html')


@app.route('/face_detection')
def face_detection_feed():
    return Response(face_detection_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
