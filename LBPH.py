import cv2


def lbph(img):
    """
     the function gets a square img of a face, execute the lbph process on it and returns the image's histogram
     and its lbph.

    """
    img = cv2.resize(img, (40, 40))
    img = LBP_process(img)
    grids = generate_grid(img)
    img_hist = concatenate_histograms(grids)
    return img_hist, img


def LBP_process(img):
    """
    the function gets an image and performs the lbp operation on it
    """
    copy_img = img.copy()
    height, width = img.shape

    for y in xrange(1, height):
        for x in xrange(1, width):

            # determining the threshold (middle pixel)
            threshold = img[y, x]
            # determining the sliding window as 3X3
            window = img[y - 1:y + 2, x - 1:x + 2].copy()

            win_height, win_width = window.shape
            #sliding over the window, performing the lbp operation

            for a in xrange(win_height):
                for b in xrange(win_width):
                    if window[a, b] < threshold:
                        window[a, b] = 0
                    else:
                        window[a, b] = 1

            # concatenate the neighbours pixels into an 8 bit number
            byte_identity = ''
            for a in xrange(win_height):
                for b in xrange(win_width):
                    byte_identity += str(window[a, b])

            # deleting the threshold character
            midlen = len(byte_identity) / 2
            byte_identity = byte_identity[:midlen] + byte_identity[midlen + 1:]

            # inserting the new value into the pixel
            copy_img[y, x] = int(byte_identity, 2)

    cv2.imshow('lbph', copy_img)
    cv2.waitKey(50)
    cv2.destroyAllWindows()
    return copy_img


def generate_grid(img, grid_x=8, grid_y=8):
    """
    the function gets an image and divide it into a grid with the received dimensions, returning a list of regions.
    explanation:
    width of an image / number of the horizontal grids = the minimum, or normal, width of a section
    width of an image % width of a section = number of sections that should be increased in size
    same goes for vertical sections.
    """

    height, width = img.shape
    #setting grid width and grid height
    grid_width = width / grid_x
    grid_height = height / grid_y

    grids = []
    last_width = 0
    last_height = 0

    for y in xrange(grid_y):
        for x in xrange(grid_x):
            # if this horizontal section should be increased
            if x < width % grid_width:
                # and this vertical section should be increased
                if y < height % grid_height:
                    # create an increased grid region by both height and width
                    grids.append(img[last_height:last_height + grid_height + 1, last_width:last_width + grid_width + 1])
                    last_width += grid_width + 1
                else:
                    # create an increased grid region only by width
                    grids.append(img[last_height:last_height + grid_height, last_width:last_width + grid_width + 1])
                    last_width += grid_width + 1
            # if this vertical section should be increased
            elif y < height % grid_height:
                # create an increased grid region only by height
                grids.append(img[last_height:last_height + grid_height + 1, last_width:last_width + grid_width])
                last_width += grid_width
            # if this section shouldn't be increased
            else:
                #create a normal grid region
                grids.append(img[last_height:last_height + grid_height, last_width:last_width + grid_width])
                last_width += grid_width
        # reset last width
        last_width = 0
        # if created an increased vertical section
        if y < height % grid_height:
            last_height += grid_height + 1
        else:
            last_height += grid_height

    return grids


def concatenate_histograms(grids):
    """
    the functions gets a list of image-grids, creates their histograms and concatenate them,
    returning the image histogram.
    """
    img_hist = []
    for grid in grids:
        hist = generate_histogram(grid)
        img_hist += hist
    return img_hist


def generate_histogram(grid):
    """
    the function gets an image and returns an histogram of its values.
    """
    flat = flatten_grid(grid)
    a = [0] * 256
    for num in flat:
        a[num] += 1

    for x in xrange(len(a)):
        a[x] = float(a[x]) / grid.size

    return a







def flatten_grid(grid):
    """
    the function transforms a multidimensional matrix into to a list
    """
    return [x for sublist in grid for x in sublist]

