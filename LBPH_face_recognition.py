import cv2
import cv3
from LBPH import *
from general_functions import *
from scipy.misc import imsave
import pickle
import numpy as np


""" The following code is meant to be executed in an environment which is structured as shown:
---code
---data
    ---test_data
    ---training_data
"""

#globals
DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TEST_DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'test_data')
TRAINING_DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'training_data')
STORED_DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'prepared_faces')
LBPH_DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'lbph_faces')
LBPH_HIST_PATH = os.path.join(DATA_FOLDER_PATH, 'lbph_hist')
STORED_FACES_PATH = os.path.join(DATA_FOLDER_PATH, 'prepared_images')

"""
===========================================
===============PREPARING===================
===========================================
"""

def store_training_data(training_data_folder_path=TRAINING_DATA_PATH):
    """
    the function gets the path of the data folder, and prepares the training data (detecting faces, cropping them),
    and saves it to the hard drive for later use.
    """
    count = 0
    # creates a list for output
    faces = []
    #get the paths of images inside the data folder
    img_paths = get_images_paths(training_data_folder_path)

    for path, label in img_paths:
        count += 1
        face = cv3.detect_faces(path, 1)
        # a face is a tuple comprised of: (the actual image of the face, (upper left corner coordinates and it size))
        #ignore images with undetected faces
        if len(face) != 0:
            new_dir_data = STORED_DATA_PATH
            new_dir_images = STORED_FACES_PATH

            if not os.path.exists(new_dir_data):
                os.makedirs(new_dir_data)

            if not os.path.exists(new_dir_images):
                os.makedirs(new_dir_images)

            new_path_data = os.path.join(new_dir_data, 's' + str(label))
            if not os.path.exists(new_path_data):
                os.makedirs(new_path_data)

            new_path_faces = os.path.join(new_dir_images, 's' + str(label))
            if not os.path.exists(new_path_faces):
                os.makedirs(new_path_faces)

            pickle.dump((face[0], label), open(os.path.join(new_path_data, str(count) + "-" + str(label)), 'wb'))
            imsave(os.path.join(new_path_faces, str(count) + "-" + str(label)+'.jpg'), face[0][0])

            #add face to list of faces
            faces.append((face[0], label))
        else:
            #deletes an image that has no face
            os.remove(path)

    return faces


def prepare_training_data(training_data_folder_path=TRAINING_DATA_PATH):
    """
    the function gets the path of the data folder, and prepares the training data (detecting faces, cropping them),
    and returns a list of faces and labels.
    """
    # creates a list for output
    faces = []
    #get the paths of images inside the data folder
    img_paths = get_images_paths(training_data_folder_path)

    for path, label in img_paths:
        face = cv3.detect_faces(path, 1)
        # a face is a tuple comprised of: (the actual image of the face, upper left corner coordinates and it size)
        #ignore images with undetected faces
        if len(face) != 0:
            #add face to list of faces
            faces.append((face[0], label))

    return faces


def load_training_data(stored_data_path=STORED_DATA_PATH):
    """
        the function gets the path of the stored-data folder, and loads it as a list of faces and labels, and return it.
    """
    faces = []
    faces_paths = get_images_paths(stored_data_path)
    for path, label in faces_paths:
        face = pickle.load(open(path))
        faces.append(face)
    return faces

"""
===========================================
================TRAINING===================
===========================================
"""


def store_train(faces):
    """
    the function gets a list of faces, and train on them, returning the histograms of the faces, storing it on hard disk
    """
    faces_hist = []
    count = 0

    for face, label in faces:
        hist, lbp_img = lbph(face[0])

        newdir = LBPH_DATA_PATH
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        newpath = os.path.join(newdir, 's' + str(label))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        imsave(os.path.join(newpath, str(count) + "-" + str(label)) + '.jpg', lbp_img)

        newdir = LBPH_HIST_PATH
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        newpath = os.path.join(newdir, 's' + str(label))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        pickle.dump((hist, label), open(os.path.join(newpath, str(count) + "-" + str(label)), 'wb'))
        faces_hist.append((hist, label))
        count += 1
    return faces_hist


def load_train(stored_lbph_path=LBPH_HIST_PATH):
    """
        the function gets the path of the stored-data folder, and loads it as a list of faces and labels, and return it.
    """
    faces_hist = []
    hist_paths = get_images_paths(stored_lbph_path)
    for path, label in hist_paths:
        hist = pickle.load(open(path))
        faces_hist.append(hist)
    return faces_hist


def train(faces):
    """
    the function gets a list of faces, and train on them, returning the histograms of the faces
    """
    faces_hist = []
    for face, label in faces:
        hist, lbp_img = lbph(face[0])
        faces_hist.append((hist, label))
    return faces_hist

"""
===========================================
==================TEST=====================
===========================================
"""


def test(training_images_hist, test_folder_path):
    """
    the function goes over the images in the test folder and recgnize the persons in them
    """
    results = []
    testing_images = os.listdir(test_folder_path)

    for image_name in testing_images:
        path = test_folder_path + "/" + image_name
        img = cv2.imread(path)
        face = cv3.detect_faces(path, 1)
        if len(face) != 0:
            result, result_name = predict(face[0], training_images_hist)
            results.append([result, result_name, image_name, str(image_name[:image_name.index('-')] == str(result_name))])
            draw_text(img, str(result_name) + " " + str(result), face[0][1][0], face[0][1][1])
            cv2.imshow(str(result_name), img)
            cv2.waitKey(1000)
        else:
            os.remove(path)
    cv2.destroyAllWindows()
    return results


def compare(img_hist1, img_hist2):
    """
    the function get two histograms and compare them, returning the euclidean  distance between them
    """

    sum = 0
    for value1, value2 in zip(img_hist1, img_hist2):
        sum += (value1 - value2) ** 2
    return sum ** 0.5


def predict(face, training_images_hist):
    """
    the function gets a face tuple: (image,size), and predicts who is the person in the image
    """
    cv2.imshow('testing', face[0])
    cv2.waitKey(100)
    img_hist, img = lbph(face[0])
    imsave(str(img_hist[0]) + '.jpg', img)
    minimum = compare(training_images_hist[0][0], img_hist)
    minmum_name = training_images_hist[0][1]
    for train_image_hist, label in training_images_hist:
        result = compare(train_image_hist, img_hist)
        if result < minimum:
            minimum = result
            minmum_name = label
    cv2.destroyAllWindows()
    return minimum, minmum_name


def main():
    print "Did u add any new images/data?(Y/N):"
    answer = raw_input()

    while answer != 'y' and answer != 'Y' and answer != 'N' and answer != 'n':
        print "Error: wrong input\n"
        print "Did u add any new images/data?(Y/N):"
        answer = raw_input()

    if answer == 'n' or answer == 'N':
        print "training data prepared"
        print "training on stored data"
        training_images_hists = load_train()
        print "testing"
        results = test(training_images_hists, TEST_DATA_PATH)
        print "results:\n"
        table_print(['Distance', 'IDentity', 'Image_name', 'Success'], results)

    elif answer == 'y' or answer == 'Y':
        print ("preparing new training data and storing it")
        faces = store_training_data()
        print ("training on new training data")
        training_images_hists = store_train(faces)
        print "testing"
        results = test(training_images_hists, TEST_DATA_PATH)
        print "results:\n"
        table_print(['Distance', 'IDentity', 'Image_name', 'Success'], results)

    else:
        print "It seems like something went wrong"


if __name__ == "__main__":
    main()
