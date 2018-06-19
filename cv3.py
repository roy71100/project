# This is a face detection module that was created by me (Roy Amir), through using already existing solutions.

import cv2


CV2_CASCADE_PATH = r"C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CV2_CASCADE_PATH)


def detect_faces(img_path, num_of_faces=-1):
    """
    the following function gets an image path and number of faces its should return and returns a list of tuples of
    all the faces it found in this image,together with the coordinates of their upper left corners and their height
    and width. if num_of_faces = -1, all detected faces will be returned.
    [(face_image,(x,y,w,h)),(face_image1,(x1,y1,w1,h1))...]
    """

    counter = 0
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    final_faces = []
    for (x, y, w, h) in faces:
        if counter >= num_of_faces != -1:
            break
        else:
            counter += 1
            cropped_face = crop_face(gray[y:y + h, x:x + w])
            final_faces.append((cropped_face, (x, y, w, h)))
    return final_faces


def crop_face(face, ratio=0.2):
    """
    the following function gets a image of a face and crops it to a certain ratio,
    returns the cropped face.
    """
    ratio /= 2
    height, width = face.shape[0], face.shape[1]
    cropped_face = face[int(ratio * height):int((1 - ratio) * height), int(width * ratio):int(width * (1 - ratio))]

    return cropped_face






