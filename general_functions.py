import cv2
import os


def color_to_grayscale(img):
    """
    the functions transforms an image to grayscale
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def draw_text(img, text, x, y):
    """
    the functions draws text on an image in a certain spot
    """
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)


def get_images_paths(data_folder_path):
    """ the function gets the directory of the data folder and return a list of paths of the files inside the directory,
     and its subdirectories (which are images) .
    """

    dirs = os.listdir(data_folder_path)
    images_paths = []

    #go through each directory and obtain images within it
    for dir_name in dirs:
        #subject directories start with 's', ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        #extract label number of subject from dir_name (format of dir name = slabel)
        label = int(dir_name.replace("s", ""))

        #build path of directory containing images for current subject, sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the names of the images that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            #build image path, sample image path = training-data/s1/1.png
            image_path = subject_dir_path + "/" + image_name
            images_paths.append((image_path, label))

    return images_paths


def table_print(headers, data):
    false_count = 0
    true_count = 0
    #find desired column size (maximum size of input to show)
    col_lens = [len(str(h)) for h in headers]
    for d in data:
        for e in d:
            if len(str(e)) > col_lens[d.index(e)]:
                col_lens[d.index(e)] = len(str(e))

    # create and print headers row
    str_head = "|"
    for header in headers:
        padding_size = col_lens[headers.index(header)] - len(str(header))
        if padding_size % 2 == 1:
            padding_size += 1
        col_lens[headers.index(header)] = padding_size + len(str(header)) + 4
        str_head += ((padding_size / 2) + 2) * ' ' + header + ((padding_size / 2) + 2) * ' ' + "|"
    print str_head
    print "-" * len(str_head)

    # print data, row by row
    str_row = '|'
    for drow in data:
        for e in drow:
            if e == 'True' or e == 'False':
                if e == 'True':
                    true_count += 1
                else:
                    false_count += 1
            cell_len = col_lens[drow.index(e)]
            padding = cell_len - len(str(e))
            str_row += (padding / 2) * " " + str(e) + (padding - (padding / 2)) * " " + "|"
        print str_row + "\n"
        str_row = '|'
    print "number of hits: {}  number of misses: {} success rates: {}".format(true_count, false_count, float(
        true_count) / (true_count + false_count))

