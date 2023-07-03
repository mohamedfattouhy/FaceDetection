# MANAGE ENVIRONMENT
import cv2
import numpy as np
import tensorflow as tf


def real_time_face_detection(model):

    cap = cv2.VideoCapture(0)
    print()
    print("Press \'q\' to quit the session")
    print()

    while cap.isOpened():

        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = model.predict(np.expand_dims(resized/255, 0))
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

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
