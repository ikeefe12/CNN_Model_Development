# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2

number_of_frames = 5207
raw_data_filename = "filename.txt"
training_data_filename = raw_data_filename.split('.')[0] + '_training.txt'

gray_16_images = np.zeros((number_of_frames, 84, 120))

img_num = 0
img_disp = 0

train_min = 0
train_max = 0

x_mouse = 0
y_mouse = 0

f_in = open(raw_data_filename)
f_out = open(training_data_filename, 'w')


def read_images():
    global img_num
    for line in f_in:
        numbers = [x for x in line.split()]
        for i in range(len(gray_16_images[img_num])):
            for j in range(len(gray_16_images[img_num][i])):
                gray_16_images[img_num][i][j] = int(numbers[i * 120 + j])
        img_num += 1
        print(img_num)


def mouse_events(event, x, y, flags, params):
    global img_disp, img_num
    if event == cv2.EVENT_MOUSEMOVE:
        global x_mouse
        global y_mouse
        x_mouse = x
        y_mouse = y
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X-Coordinate:" + str(x_mouse))
        print("Y-Coordinate:" + str(y_mouse))
    if event == cv2.EVENT_RBUTTONDOWN:
        write_output(x_mouse, y_mouse)
        if img_disp < img_num - 1:
            img_disp += 1
            if img_disp % 100:
                print(img_disp)
        else:
            f_out.close()
            exit(0)


def write_output(x, y):
    global img_disp
    str_out = gray_16_images[img_disp][:][:]
    str_out = (str_out - train_min) / (train_max - train_min)
    # str_out = str_out.astype('float32') / 65535
    str_out = str_out.flatten()
    str_out = np.append(str_out, x / 119)
    str_out = np.append(str_out, y / 83)
    for el in str_out:
        f_out.write(str(el))
        f_out.write(' ')
    f_out.write('\n')


# Main Functionality
read_images()
train_min = np.min(gray_16_images)
train_max = np.max(gray_16_images)
print(train_min)
print(train_max)
cv2.namedWindow("Inferno", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Inferno", 720, 504)
while 1:
    gray8_image = np.zeros((84, 120), dtype=np.uint8)
    gray8_image = cv2.normalize(gray_16_images[img_disp][:][:], gray8_image, 0, 255, cv2.NORM_MINMAX)
    gray8_image = np.uint8(gray8_image)
    inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
    cv2.circle(inferno_palette, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
    cv2.imshow("Inferno", inferno_palette)
    cv2.setMouseCallback('Inferno', mouse_events)
    cv2.waitKey(500)
