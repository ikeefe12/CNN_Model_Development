# import keras
from keras.models import load_model
import numpy as np
import cv2

gray_16_images = np.zeros((5207, 84, 120))

img_num = 0
img_disp = 250

x_mouse = 0
y_mouse = 0

f_in = open('8_min_training_video.txt')


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


min_train = 2657.0
max_train = 2884.0

# Load the model from the .h5 file
model = load_model('cnn_model.h5')
challenger_model = load_model('cnn_compare.h5')

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
    [[x_chall, y_chall]] = challenger_model.predict(predic_data)

    gray8_image = np.zeros((84, 120), dtype=np.uint8)
    gray8_image = cv2.normalize(raw_image, gray8_image, 0, 255, cv2.NORM_MINMAX)
    gray8_image = np.uint8(gray8_image)

    inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
    cv2.circle(inferno_palette, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
    cv2.circle(inferno_palette, (int(x_curr * 119), int(y_curr * 83)), 1, (0, 0, 0), -1)
    cv2.circle(inferno_palette, (int(x_chall * 119), int(y_chall * 83)), 1, (255, 255, 255), -1)
    cv2.imshow("Inferno", inferno_palette)
    cv2.setMouseCallback('Inferno', mouse_events)
    cv2.waitKey(100)
