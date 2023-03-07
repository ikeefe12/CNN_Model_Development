import socket
import select
import struct
import binascii

import cv2
import numpy
import time

# IP_ADDRESS = "192.168.0.120"
IP_ADDRESS = "192.168.0.120"
PORT = 30444


def format_data(data_binary):
    hex_str = binascii.hexlify(data_binary)
    numbers = [hex_str[i:i + 4] for i in range(0, len(hex_str), 4)]
    numbers = [int(x, 16) for x in numbers]
    return numbers

def find_hottest_center(image):  # image in grey 8 scale
    ret, thresh = cv2.threshold(image, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    max_avg_temp = 0
    for i, cnt in enumerate(contours):
        mask = numpy.zeros_like(image)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        avg_temp = numpy.mean(numpy.where(mask == 255, image, 0))
        if avg_temp > max_avg_temp:
            max_avg_temp = avg_temp
            index = i
    cnt = contours[index]
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    label = (cX, cY)
    return label


# Create a socket and connect to the ESP32 device
print("opening socket")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("connecting to socket")
client_socket.connect((IP_ADDRESS, PORT))

print("send data: BIND")
# Send some data to the device
client_socket.send(b"Bind HTPA series device")

data = client_socket.recv(1024)
print(data)

# Send some data to the device
print("send data: TRIGGER")
client_socket.send(b"K")

print("Start Receiving THERMAL IMAGE DATA")

cv2.namedWindow("Inferno", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Inferno", 720, 504)

while True:
    client_socket.send(b"N")
    data = b''

    # set the number of packets to receive
    num_packets = 21

    # set a counter for the number of packets received
    packets_received = 0

    start_time = time.time()
    # loop until the required number of packets have been received
    while packets_received < num_packets:
        print(packets_received)
        # if more than half a second passes, likely a UDP packet was lost - get next frame
        if (time.time() - start_time) > 0.5:
            print("Missed frame")
            break
        try:
            # receive a packet from the socket (with a maximum size of 1024 bytes)
            new_data, address = client_socket.recvfrom(961)
            # add the packet to the list of received packets
            data += new_data[1:]
            # increment the counter for the number of packets received
            packets_received += 1
        except socket.error:
            # if the socket.recv function times out, continue to the next iteration of the loop
            continue

    if packets_received == 21:
        grey_16 = format_data(data)
        grey_16 = numpy.array(grey_16).reshape(84, 120)

        gray8_image = numpy.zeros((84, 120), dtype=numpy.uint8)
        gray8_image = cv2.normalize(grey_16[:][:], gray8_image, 0, 255, cv2.NORM_MINMAX)
        gray8_image = numpy.uint8(gray8_image)
        inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)

        [x_con, y_con] = find_hottest_center(gray8_image)
        cv2.circle(inferno_palette, (int(x_con), int(y_con)), 1, (255, 255, 255), -1)

        cv2.imshow("Inferno", inferno_palette)
        cv2.waitKey(50)



# Close the socket
client_socket.close()
