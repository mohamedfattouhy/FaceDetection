# MANAGE ENVIRONMENT
import os
import time
import uuid
import cv2


def collect_data(save_path: str, n_images: int) -> None:
    """Collect images of your face with your webcam"""

    cap = cv2.VideoCapture(0)

    for imgnum in range(n_images):

        print('Collecting image {}'.format(imgnum))
        _, frame = cap.read()
        img_name = os.path.join(save_path, f'{str(uuid.uuid1())}.jpg')

        cv2.imwrite(img_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(5)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
