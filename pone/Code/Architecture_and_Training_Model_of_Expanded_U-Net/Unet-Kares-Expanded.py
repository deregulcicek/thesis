import tensorflow as tf
import numpy as np
import random
import os
import sys
import pickle
from PIL import Image 
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from skimage import io
from skimage.transform import resize


#Training

input_name = os.listdir('/Users/kunthar/development/gulcicek/patient_data/marked')
output_name = os.listdir('/Users/kunthar/development/gulcicek/patient_data/unmarked')
n = len(input_name)
batch_size = 8
input_size_1 = 256
input_size_2 = 256

def batch_data(input_name, n, batch_size = 8, input_size_1 = 256, input_size_2 = 256):
    rand_num = random.randint(0,n-1)
    
    img1 = io.imread('/Users/kunthar/development/gulcicek/patient_data/marked/'+input_name[rand_num]).astype("float")
    img2 = io.imread('/Users/kunthar/development/gulcicek/patient_data/unmarked/'+output_name[rand_num]).astype("float")
    img1 = resize(img1, [input_size_1, input_size_2, 3])
    img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
    img1 /= 255
    img2 /= 255 
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n-1)
        img1 = io.imread('/Users/kunthar/development/gulcicek/patient_data/marked/'+input_name[rand_num]).astype("float")
        img2 = io.imread('/Users/kunthar/development/gulcicek/patient_data/unmarked/'+output_name[rand_num]).astype("float")
        img1 = resize(img1, [input_size_1, input_size_2, 3])
        img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis = 0)
        batch_output = np.concatenate((batch_output, img2), axis = 0)
    return batch_input, batch_output


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

inpt = Input(shape=(input_size_1, input_size_2, 3))

conv1 = Conv2d_BN(inpt, 8, (3, 3))
conv1 = Conv2d_BN(conv1, 8, (3, 3))
pool1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv1)

conv2 = Conv2d_BN(pool1, 16, (3, 3))
conv2 = Conv2d_BN(conv2, 16, (3, 3))
pool2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv2)

conv3 = Conv2d_BN(pool2, 32, (3, 3))
conv3 = Conv2d_BN(conv3, 32, (3, 3))
pool3 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv3)

conv4 = Conv2d_BN(pool3, 64, (3, 3))
conv4 = Conv2d_BN(conv4, 64, (3, 3))
pool4 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv4)

conv5 = Conv2d_BN(pool4, 128, (3, 3))
conv5 = Dropout(0.5)(conv5)
conv5 = Conv2d_BN(conv5, 128, (3, 3))
conv5 = Dropout(0.5)(conv5)

convt1 = Conv2dT_BN(conv5, 64, (3, 3))
concat1 = concatenate([conv4,convt1], axis=3)
concat1 = Dropout(0.5)(concat1)
conv6 = Conv2d_BN(concat1, 64, (3, 3))
conv6 = Conv2d_BN(conv6, 64, (3, 3))

convt2 = Conv2dT_BN(conv6, 32, (3, 3))
concat2 = concatenate([conv3, convt2], axis=3)
concat2 = Dropout(0.5)(concat2)
conv7 = Conv2d_BN(concat2, 32, (3, 3))
conv7 = Conv2d_BN(conv7, 32, (3, 3))

convt3 = Conv2dT_BN(conv7, 16, (3, 3))
concat3 = concatenate([conv2, convt3], axis=3)
concat3 = Dropout(0.5)(concat3)
conv8 = Conv2d_BN(concat3, 16, (3, 3))
conv8 = Conv2d_BN(conv8, 16, (3, 3))

convt4 = Conv2dT_BN(conv8, 8, (3, 3))
concat4 = concatenate([conv1, convt4], axis=3)
concat4 = Dropout(0.5)(concat4)
conv9 = Conv2d_BN(concat4, 8, (3, 3))
conv9 = Conv2d_BN(conv9, 8, (3, 3))
conv9 = Dropout(0.5)(conv9)
outpt = Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='same', activation='sigmoid')(conv9)

model = Model(inpt, outpt)
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['acc'])
model.summary()
itr = 50
S = []
for i in range(itr):
    print("iteration = ", i+1)
    if i < 500:
        bs = 4
    elif i < 2000:
        bs = 8
    elif i < 5000:
        bs = 16
    else:
        bs = 32
    train_X, train_Y = batch_data(input_name, n, batch_size = 16)
    history = model.fit(train_X, train_Y, epochs=1, verbose=2)
    #if i % 100 == 99:
save_model(model, '/Users/kunthar/development/gulcicek/pone/Code/Architecture_and_Training_Model_of_Expanded_U-Net/dombili.h5')



#TEST

model = load_model('/Users/kunthar/development/gulcicek/pone/Code/Architecture_and_Training_Model_of_Expanded_U-Net/dombili.h5')

test_input_name = os.listdir('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-BreastTumourImages')
test_output_name = os.listdir('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-3-channel-Labels')
n_test = len(test_input_name)

def batch_data_test(input_name, n, batch_size = 4, input_size_1 = 256, input_size_2 = 256):
    rand_num = random.randint(0, n-1)
    img1 = io.imread('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-BreastTumourImages/'+test_input_name[rand_num]).astype("float")
    img2 = io.imread('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-3-channel-Labels/'+test_output_name[rand_num]).astype("float")
    img1 = resize(img1, [input_size_1, input_size_2, 3])
    img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n-1)
        img1 = io.imread('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-BreastTumourImages/'+test_input_name[rand_num]).astype("float")
        img2 = io.imread('/Users/kunthar/development/gulcicek/patient_data/Test-Expanded-3-channel-Labels/'+test_output_name[rand_num]).astype("float")
        img1 = resize(img1, [input_size_1, input_size_2, 3])
        img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis = 0)
        batch_output = np.concatenate((batch_output, img2), axis =0)
    return batch_input, batch_output

test_X, test_Y = batch_data_test(test_input_name, n_test, batch_size = 1)
pred_Y = model.predict(test_X)

accur = history.history['acc']
epochs = range(1, len(accur)+1)
plt.plot(epochs, accur, label = 'Training acc')
plt.title('Training loss and acc')
plt.xlabel('Epochs')
plt.ylabel('loss or acc')
plt.legend()

loss = history.history['loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs, loss, label = 'Training loss')
plt.title('Training loss and acc')
plt.xlabel('Epochs')
plt.ylabel('loss or acc')
plt.legend()

ii = 0
plt.figure()
plt.imshow(test_X[ii, :, :, :])
plt.axis('off')
plt.imsave('/Users/kunthar/development/gulcicek/pone/Segmentation_Results/Expanded_U-Net/1/1.jpg',test_X[ii, :, :, :])
plt.figure()
plt.imshow(test_Y[ii, :, :, :])
plt.axis('off')
plt.imsave('/Users/kunthar/development/gulcicek/pone/Segmentation_Results/Expanded_U-Net/1/2.jpg',test_Y[ii, :, :, :])
plt.figure()
plt.imshow(pred_Y[ii, :, :, :])
plt.axis('off')
plt.imsave('/Users/kunthar/development/gulcicek/pone/Segmentation_Results/Expanded_U-Net/1/3.jpg',pred_Y[ii, :, :, :])
plt.show()
sys.exit(0)

    
# input_single_x = os.listdir('H:/segmentation_experiment/3channel_3class_255pixel/single_input')
# n_single = len(input_single_x)

# rand_num_single = random.randint(0, n_single-1)

# input_single_x = io.imread('H:/segmentation_experiment/3channel_3class_255pixel/single_input/'+input_single_x[rand_num_single]).astype("float")
# input_single_x = resize(input_single_x, [input_size_1, input_size_2, 3])
# input_single_x = np.reshape(input_single_x, (1, input_size_1, input_size_2, 3))
# input_single_x /= 255

# pred_single_Y = model.predict(input_single_x)
# jj = 0
# plt.figure()
# plt.imshow(input_single_x[jj, :, :])
# plt.axis('off')
# plt.figure()
# plt.imshow(pred_single_Y[jj, :, :])
# plt.axis('off')
# plt.show()
# sys.exit(0)


