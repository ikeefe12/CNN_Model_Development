# import keras
from keras.models import load_model
import numpy as np
import cv2

test_data_filename = '8_min_training_video.TXT'

gray_16_images = np.zeros((5207, 84, 120))

img_num = 0
img_disp = 250

x_mouse = 0
y_mouse = 0

f_in = open(test_data_filename)


# this is the simple algorithm that the CNN must outperform in accuracy and or prediction time
def find_center_of_heat(thermal_image, threshold=0.98):
    # Find the maximum value in the image
    max_val = np.max(thermal_image)
    coordinates = []

    # Iterate through all pixels in the image
    for y in range(thermal_image.shape[0]):
        for x in range(thermal_image.shape[1]):
            # Check if the pixel is above the threshold
            if thermal_image[y, x] >= threshold * max_val:
                coordinates.append([x, y])
    # calculate the center position
    center_x = np.round(np.mean([i[0] for i in coordinates]))
    center_y = np.round(np.mean([i[1] for i in coordinates]))
    return center_x, center_y


def find_hottest_center(image):  # image in grey 8 scale
    ret, thresh = cv2.threshold(image, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    max_avg_temp = 0
    for i, cnt in enumerate(contours):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        avg_temp = np.mean(np.where(mask == 255, image, 0))
        if avg_temp > max_avg_temp:
            max_avg_temp = avg_temp
            index = i
    cnt = contours[index]
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    label = (cX, cY)
    return label


def read_images():
    global img_num
    for line in f_in:
        numbers = [x for x in line.split()]
        for i in range(len(gray_16_images[img_num])):
            for j in range(len(gray_16_images[img_num][i])):
                gray_16_images[img_num][i][j] = int(numbers[i * 120 + j])
        img_num += 1


def mouse_events(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        global x_mouse
        global y_mouse
        x_mouse = x
        y_mouse = y
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X-Coordinate:" + str(x_mouse))
        print("Y-Coordinate:" + str(y_mouse))
    if event == cv2.EVENT_RBUTTONDOWN:
        exit(0)


min_train = 2636.0
max_train = 2900.0

# Load the model from the .h5 file
model = load_model('cnn_model.h5')
# challenger_model = load_model('cnn_compare.h5')

# Load the training data from the .txt file
read_images()

cv2.namedWindow("Inferno", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Inferno", 720, 504)

while 1:
    if img_disp < img_num - 1:
        img_disp += 1
    else:
        exit(0)

    raw_image = gray_16_images[img_disp][:][:]
    predic_data = (raw_image - min_train) / (max_train - min_train)
    predic_data = predic_data.reshape(1, 120, 84, 1)

    [[x_curr, y_curr]] = model.predict(predic_data)
    # [[x_chall, y_chall]] = challenger_model.predict(predic_data)
    # [x_simp, y_simp] = find_center_of_heat(raw_image)

    gray8_image = np.zeros((84, 120), dtype=np.uint8)
    gray8_image = cv2.normalize(raw_image, gray8_image, 0, 255, cv2.NORM_MINMAX)
    gray8_image = np.uint8(gray8_image)
    [x_con, y_con] = find_hottest_center(gray8_image)

    inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
    cv2.circle(inferno_palette, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
    cv2.circle(inferno_palette, (int(x_curr * 119), int(y_curr * 83)), 1, (0, 0, 0), -1)
    cv2.circle(inferno_palette, (int(x_con), int(y_con)), 1, (255, 255, 255), -1)
    cv2.imshow("Inferno", inferno_palette)
    cv2.setMouseCallback('Inferno', mouse_events)
    cv2.waitKey(100)
