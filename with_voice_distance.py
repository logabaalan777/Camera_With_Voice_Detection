import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import pyttsx3

url = 'http://172.31.99.211/cam-hi.jpg'
im = None

# Initialize the pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def calculate_distance(pixel_width, known_width, focal_length):
    # Calculate the distance to the object based on its width, known width, and focal length
    return (known_width * focal_length) / pixel_width

def say_distance(distance):
    # Convert the distance value to speech
    engine.say(f'The distance is {distance:.2f} meters')
    engine.runAndWait()

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        cv2.imshow('live transmission', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    # Set the known width of the object in the scene
    known_width = 0.2  # Example: width of a standard-size credit card

    # Set the focal length of the camera
    # You can determine the focal length experimentally or use a calibration process
    focal_length = 500  # Example value

    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)

        for box, lbl in zip(bbox, label):
            x, y, width, height = box
            distance = calculate_distance(width, known_width, focal_length)
            cv2.putText(im, f'{distance:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            say_distance(distance)

        cv2.imshow('detection', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

        # Convert the labels to a single string
        object_labels = ', '.join(label)

        # Convert the labels to speech
        engine.say(object_labels)
        engine.runAndWait()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = executor.submit(run1)
        f2 = executor.submit(run2)
