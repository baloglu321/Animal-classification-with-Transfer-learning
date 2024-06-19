import cv2 as cv
import numpy as np

EXTENSIONS = ("jpg", "JPG", "jpeg", "JPEG", "png", "PNG")


def detect(pil_image, square=False):
    # img = cv.imread(image_path)
    img = np.array(pil_image)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create(edgeThreshold=8)
    kp = sift.detect(gray, None)

    all_points = [i.pt for i in kp]
    x_points = [z[0] for z in all_points]
    y_points = [z[1] for z in all_points]
    if len(x_points) == 0 or len(y_points) == 0:
        return pil_image
    thresh = 0
    x_min, y_min = int(min(x_points)) - thresh, int(min(y_points) - thresh)
    x_max, y_max = int(max(x_points)) + thresh, int(max(y_points) + thresh)
    min_side = min((x_max - x_min), (y_max - y_min))
    max_side = max((x_max - x_min), (y_max - y_min))
    x_mean, y_mean = int((x_max + x_min) / 2), int((y_max + y_min) / 2)
    # img = cv.drawKeypoints(img, kp, img)
    squared_x_min, squared_x_max = x_mean - int(min_side / 2), x_mean + int(
        min_side / 2
    )
    squared_y_min, squared_y_max = y_mean - int(min_side / 2), y_mean + int(
        min_side / 2
    )

    if not square:
        return img[y_min:y_max, x_min:x_max]

    elif square:
        return img[squared_y_min:squared_y_max, squared_x_min:squared_x_max]
