# MANAGE ENVIRONMENT
import json
import cv2


def display_rectangle_detection(image_path, label_path):

    img_sample = cv2.imread(image_path)
    print()
    print("Image shape:", img_sample.shape)

    # cv2.imshow("Sample Image", img_sample)
    # cv2.waitKey(0)

    with open(label_path, 'r') as f:
        label = json.load(f)

    # print(tuple(label['shapes']))

    cv2.rectangle(img=img_sample,
                  pt1=(int(label['shapes'][0]['points'][0][0]),
                       int(label['shapes'][0]['points'][0][1])),
                  pt2=(int(label['shapes'][0]['points'][1][0]),
                       int(label['shapes'][0]['points'][1][1])),
                  color=(255, 0, 0),
                  thickness=1)

    label_img = 'Face Detected'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_origin = (int(label['shapes'][0]['points'][0][0]),
                   int(label['shapes'][0]['points'][0][1]) - 10)

    cv2.putText(img_sample, label_img, org=text_origin,
                fontFace=font, color=(255, 0, 0),
                fontScale=0.6)

    cv2.imshow("Sample Image Detection", img_sample)
    cv2.waitKey(0)
