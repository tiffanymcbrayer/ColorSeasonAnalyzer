import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical

import facial_features as ff

data = 'targets.txt'
def read_data():
    patterns = []
    targets = []
    images = glob.glob("./ColorCorrectedImages/*.jpg")
    valid_targets = ['0','1','2','3']
    f = open(data,'r')
    count = 0
    f2 = open('patterns.txt','r')
    f2_lines = f2.readlines()
    for line in f.readlines():
        if count == 0:
            pass
        else:
            l = line.split()
            if len(l) > 1:
                id = int(l[0])
                # t = l[1]
                # x = l[2]
                x = l[1]
                if x in valid_targets:
                    targets.append(int(x))
                    #get patterns
                    patterns.append(f2_lines[id].split())
                    
        count += 1

    
    training_set = patterns[:int(.75*len(targets))]
    testing_set = patterns[int(.75*len(targets)):]
    training_targets = targets[:int(.75*len(targets))]
    testing_targets = targets[int(.75*len(targets)):]
    
    train_data = np.array(training_set)
    train_targets = np.array(training_targets)
    train_targets = to_categorical(train_targets)

    test_data = np.array(testing_set)
    test_targets = np.array(testing_targets)
    test_targets = to_categorical(test_targets)   

    return train_data, train_targets, test_data, test_targets


def nn():
    train_data, train_targets, test_data, test_targets = read_data()
    network = Sequential()
    network.add(Flatten(input_shape=(13,)))
    network.add(Dense(10, activation='sigmoid', name='hidden', input_shape=(13,)))
    network.add(Dense(4, activation='sigmoid', name='output'))
    network.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    network.summary()
    network.evaluate(test_data,test_targets)

def get_pattern(image):
    f = ff.detect_facial_landmarks(image)
    #undertone
    avgUndertone = 0
    for i in range(0,3):
        img = f[i+3]      
        underTone = ff.under_tone(img)
        avgUndertone += underTone
    avgUndertone = avgUndertone/3

    #eyes
    eye_color, l_eye, a_eye, b_eye, mask = ff.find_iris(f[0])
    print(eye_color)

    #hair
    hair_color, l_hair, a_hair, b_hair, mask = ff.getHair(image)
    return [avgUndertone, eye_color[0], eye_color[1], eye_color[2], l_eye, a_eye, b_eye, hair_color[0], hair_color[1], hair_color[2], l_hair, a_hair, b_hair]

def write_pattern():
    file = open('patterns.txt','w')
    for image in glob.glob("./ColorCorrectedImages/*.jpg"):
        p = get_pattern(image)
        for item in p:
            file.write(str(item) + " ")
        file.write('\n')
    file.close()


def plot_history(history):
    loss_values = history.history['loss']
    accuracy_values = history.history['accuracy']
    validation = 'val_loss' in history.history
    if validation:
        val_loss_values = history.history['val_loss']
        val_accuracy_values = history.history['val_accuracy']
    epoch_nums = range(1, len(loss_values)+1)
    plt.figure(figsize=(12,4)) # width, height in inches
    plt.subplot(1, 2, 1)
    if validation:
        plt.plot(epoch_nums, loss_values, 'r', label="Training loss")
        plt.plot(epoch_nums, val_loss_values, 'r--', label="Validation loss")
        plt.title("Training/validation loss")
        plt.legend()
    else:
        plt.plot(epoch_nums, loss_values, 'r', label="Training loss")
        plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    if validation:
        plt.plot(epoch_nums, accuracy_values, 'b', label='Training accuracy')
        plt.plot(epoch_nums, val_accuracy_values, 'b--', label='Validation accuracy')
        plt.title("Training/validation accuracy")
        plt.legend()
    else:
        plt.plot(epoch_nums, accuracy_values, 'b', label='Training accuracy')
        plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()