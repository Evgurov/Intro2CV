from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2 as cv
import skimage
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st

#===========================================
#Calculating masks for different colors

def mask_red(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower = np.array([175, 170, 100])
    upper = np.array([185, 240, 240])
    mask = cv.inRange(img_hsv, lower, upper)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    return mask_3

def mask_green(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower = np.array([73, 150, 50])
    upper = np.array([83, 230, 140])
    mask = cv.inRange(img_hsv, lower, upper)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    return mask_3

def mask_yellow(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower = np.array([17, 150, 150])
    upper = np.array([27, 220, 220])
    mask = cv.inRange(img_hsv, lower, upper)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    return mask_3

def mask_blue(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower = np.array([90, 100, 70])
    upper = np.array([120, 255, 150])
    mask = cv.inRange(img_hsv, lower, upper)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    return mask_3

def mask_black(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([200, 80, 30])
    mask = cv.inRange(img_hsv, lower, upper)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    return mask_3
#=======================================================

#=======================================================
#Count number of trains for cpecific colors

def count_trains(img, color):
    if color == "red":
        mask = mask_red(img)
        trains_img = cv.bitwise_and(img, mask)

        erosion_iterations = 0
        dilation_iterations = 3

        kernel_erosion = np.ones((3,3),np.uint8)
        kernel_dilation = np.ones((4,4),np.uint8)

        area = 5000

    elif color == "green":
        mask = mask_green(img)
        trains_img = cv.bitwise_and(img, mask)

        erosion_iterations = 0
        dilation_iterations = 3

        kernel_erosion = np.ones((4,4),np.uint8)
        kernel_dilation = np.ones((4,4),np.uint8)

        area = 5000

    elif color == "yellow":
        mask = mask_yellow(img)
        trains_img = cv.bitwise_and(img, mask)

        erosion_iterations = 1
        dilation_iterations = 3

        kernel_erosion = np.ones((3,3),np.uint8)
        kernel_dilation = np.ones((4,4),np.uint8)

        area = 4100

    elif color == "blue":
        mask = mask_blue(img)
        trains_img = cv.bitwise_and(img, mask)

        erosion_iterations = 1
        dilation_iterations = 3

        kernel_erosion = np.ones((3,3),np.uint8)
        kernel_dilation = np.ones((4,4),np.uint8)

        area = 4900

    elif color == "black":
        img = cv.blur(img, (6, 6))
        mask = mask_black(img)
        trains_img = cv.bitwise_and(img, mask)

        erosion_iterations = 1
        dilation_iterations = 3

        kernel_erosion = np.ones((3,3),np.uint8)
        kernel_dilation = np.ones((4,4),np.uint8)

        area = 4000

    trains_erosed = cv.erode(trains_img, kernel_erosion, iterations = erosion_iterations)
    trains_dilated = cv.dilate(trains_erosed, kernel_dilation, iterations = dilation_iterations)

    trains_contours, hierarchy = cv.findContours(trains_dilated[:, :, 2], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    trains_number = 0
    for contour in trains_contours:
        contour_area = cv.contourArea(contour=contour)
        trains_number += contour_area // area
        
    return trains_number
#===============================================================================

#====================================================================================
#Number of trains for every color

def trains(img):
    n_trains = {"blue": 0, "green": 0, "black": 0, "yellow": 0, "red":0}
    n_trains["blue"] = count_trains(img, "blue")
    n_trains["green"] = count_trains(img, "green")
    n_trains["black"] = count_trains(img, "black")
    n_trains["yellow"] = count_trains(img, "yellow")
    n_trains["red"] = count_trains(img, "red")
    return n_trains
#=============================================================================


#==================================================================================
#Finding city centers

def get_local_centers(corr, th):
    lbl, n = skimage.measure.label(corr >= th, connectivity=2, return_num=True)
    return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])

def find_centers(img):
    img_all_rgb = cv.cvtColor(cv.imread("./train/all.jpg"), cv.COLOR_BGR2RGB)
    template = img_all_rgb[1200:1245, 1655:1700, :]

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img_val = img_hsv[:,:,2]

    template_hsv = cv.cvtColor(template, cv.COLOR_RGB2HSV)
    template_val = template_hsv[:,:,2]
    matching = skimage.feature.match_template(img_val, template_val, pad_input=True)
    points = get_local_centers(matching, 0.7)
    return points
#=================================================================================

#=========================================================================
#Calculating scores

def get_scores(trains):
    return {x: 1.5 * y for x,y in trains.items()}

#===========================================================================

def predict_image(img: np.ndarray):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_rgb[:110, :, :] = 255
    img_rgb[:, :120, :] = 255
    img_rgb[-120:, :, :] = 255
    img_rgb[:, -120:, :] = 255

    city_centers = np.int64(find_centers(img_rgb))
    n_trains = trains(img_rgb)
    scores = get_scores(trains = n_trains)
    return city_centers, n_trains, scores
