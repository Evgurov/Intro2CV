import joblib
import cv2 as cv
import numpy as np

def preprocess(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blurred = cv.blur(img_gray, (9, 9), 0)
    img_thresholded = cv.adaptiveThreshold(img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    img_thresholded_inversed = cv.bitwise_not(img_thresholded, img_thresholded)

    erosion_kernel = np.ones((3,3), dtype=np.uint8)
    dilation_kernel = np.ones((3,3), dtype=np.uint8)
    erosion_iterations = 0
    dilation_iterations = 1
    img_thresholded_inversed = cv.erode(img_thresholded_inversed, kernel = erosion_kernel, iterations=erosion_iterations)
    img_thresholded_inversed = cv.dilate(img_thresholded_inversed, kernel = dilation_kernel, iterations=dilation_iterations)

    return img_thresholded_inversed

def get_sudoku_mask(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
    contours_areas = [cv.contourArea(contour) for contour in contours_sorted]

    sudoku_area = contours_areas[0]
    for area in contours_areas:
        if area >= sudoku_area / 1.4:
            sudoku_area = area
        else:
            sudoku_area = area
            break
    
    big_contours = [contour for contour in contours if cv.contourArea(contour) > sudoku_area]
    sudoku_contours = []

    for contour in big_contours: 
        approx_contour = cv.approxPolyDP(contour, cv.arcLength(contour, True) * 0.05, True)
        if len(approx_contour) == 4:
            (x, y, w, h) = cv.boundingRect(approx_contour)
            if w / h > 0.9 and w / h < 1.1:
                sudoku_contours.append(approx_contour)
            elif len(approx_contour) == 5:
                sudoku_contours.append(approx_contour)

    mask = np.zeros_like(img) 
    mask = cv.fillPoly(mask, sudoku_contours, 255)

    return mask, sudoku_contours

def reorder_corners(corners):
        s = corners.sum(axis = 1)
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        diff = np.diff(corners, axis = 1)
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]
        return tl, tr, br, bl

def get_sudoku_proj(img, sudoku_contours):
    sudoku_projections = []
    for contour in sudoku_contours:
        corners = np.array([[corner[0][0], corner[0][1]] for corner in contour])
        tl, tr, br, bl = reorder_corners(corners)
        clockwise_corners = np.array([tl, tr, br, bl])

        height = max(int(np.linalg.norm(tl - bl, ord=2)), int(np.linalg.norm(tr-br, ord=2)))
        width  = max(int(np.linalg.norm(tl - tr, ord=2)), int(np.linalg.norm(bl-br, ord=2)))
        transformed_corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

        transformation_matrix = cv.getPerspectiveTransform(np.array(clockwise_corners, dtype=np.float32), transformed_corners)
        projection = cv.warpPerspective(img, transformation_matrix, (width, height))

        sudoku_projections.append(projection)

    return sudoku_projections

def table_preprocess(img):
    img = cv.erode(img, np.ones((4,4)), iterations=2)
    img = cv.dilate(img, np.ones((3,3)), iterations=3)
    threshold, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV & cv.THRESH_OTSU)
    
    return img

def predict_digit(digit_img):
    return 0

def get_digits(img):
    cell_width = img.shape[0] // 9
    cell_height = img.shape[1] // 9

    sudoku_digits = [[-1 for i in range(9)] for j in range(9)]

    for i in range(9):
        for j in range(9):
            x_ul, y_ul = j * cell_width, i * cell_height
            cell_img = img[y_ul:y_ul+cell_height, x_ul:x_ul+cell_width]
            cell_img = cell_img[cell_height//10:-cell_height//10, cell_width//10:-cell_width//10]
            contours,hierarchy = cv.findContours(cell_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                for contour in contours:
                    x,y,w,h = cv.boundingRect(contour)
                    if (h > cell_height // 2) and (w > cell_width // 5):    
                        digit_img = cell_img[y-5:y+h+5, x-5:x+w+5]
                        digit_img = cv.resize(digit_img,(28,28))
                        digit_img = digit_img.reshape(1, 28, 28, 1)
                        digit_img = digit_img / 255
                        prediction = predict_digit(digit_img)
                        digit = np.argmax(prediction)
                        sudoku_digits[i][j] = digit
    
    result = []
    result.append(np.int16(sudoku_digits))

    return result


def predict_image(image: np.ndarray):
    image = preprocess(image)
    sudoku_digits = [
        np.int16([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, -1,  8,  9,  4, -1, -1, -1],
                  [-1, -1, -1,  6, -1,  1, -1, -1, -1],
                  [-1,  6,  5,  1, -1,  9,  7,  8, -1],
                  [-1,  1, -1, -1, -1, -1, -1,  3, -1],
                  [-1,  3,  9,  4, -1,  5,  6,  1, -1],
                  [-1, -1, -1,  8, -1,  2, -1, -1, -1],
                  [-1, -1, -1,  9,  1,  3, -1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    ]
    mask, sudoku_contours = get_sudoku_mask(image)

    masked_image = cv.bitwise_and(mask, image)

    sudoku_projections = get_sudoku_proj(masked_image, sudoku_contours)

    sudoku_digits = []

    for projection in sudoku_projections:
        sudoku_digits.append(get_digits(projection))

    # loading train image:
    #train_img_4 = cv.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    #rf = joblib.load('/autograder/submission/random_forest.joblib')

    return mask, sudoku_digits