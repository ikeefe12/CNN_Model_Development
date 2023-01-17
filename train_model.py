import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Load the training data from the .txt file
train_data1 = np.loadtxt('train_data.txt')
train_data2 = np.loadtxt('first_floor_data_training.txt')

train_data = np.vstack((train_data1, train_data2))
print(train_data)

# Split the data into input (pixel data) and output (labels)
x_train = train_data[:, :10080]
y_train = train_data[:, 10080:]

# Reshape the input data to match the shape of an image (120x84)
x_train = x_train.reshape(x_train.shape[0], 120, 84, 1)

# Create a sequential model
model = Sequential()

# Add a 2D convolutional layer
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(120, 84, 1)))

# Add a Max Pooling layer
model.add(MaxPooling2D(pool_size=(3, 3)))

# Add a second 2D convolutional layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Add a flatten layer
model.add(Flatten())

# Add a dense layer
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

# Train the model on the training data
model.fit(x_train, y_train, epochs=50, batch_size=64)

# Save the model
model.save('cnn_compare.h5')
