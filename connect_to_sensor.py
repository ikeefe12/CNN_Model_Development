import socket
import struct
import binascii

import cv2
import numpy

IP_ADDRESS = "192.168.0.120"
PORT = 30444


def format_data(data_binary):
    print(len(data_binary))
    hex_str = binascii.hexlify(data_binary)
    numbers = [hex_str[i:i + 4] for i in range(0, len(hex_str), 4)]
    numbers = [int(x, 16) for x in numbers]
    return numbers


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

data = b''
for x in range(14):
    # Receive data from the device
    new_data = client_socket.recv(1400)
    data += new_data

new_data = client_socket.recv(560)
data += new_data

grey_16 = format_data(data)
grey_16 = numpy.array(grey_16).reshape(120, 84)

gray8_image = numpy.zeros((84, 120), dtype=numpy.uint8)
gray8_image = cv2.normalize(grey_16[:][:], gray8_image, 0, 255, cv2.NORM_MINMAX)
gray8_image = numpy.uint8(gray8_image)
inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
cv2.imshow("Inferno", inferno_palette)
cv2.waitKey(10000)

# Close the socket
client_socket.close()
