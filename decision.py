import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import facial_features as ff

data = 'targets.txt'
def read_data(r=True):
    patterns = []
    targets = []
    images = glob.glob("./ColorCorrectedImages/*.jpg")
    valid_targets = ['0','1','2','3']
    f = open(data,'r')
    count = 0
    f2 = open('patterns3.txt','r')
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
                    # avgUndertone, eye_color_r, eye_color_g, eye_color_b, l_eye, a_eye, b_eye, hair_color_r, hair_color_g, hair_color_b, l_hair, a_hair, b_hair = f2_lines[id].split()
                    # skin_L, skin_A, skin_B, eye_r, eye_g, eye_b, eye_L, eye_A, eye_B, hair_L, hair_A, hair_B, hair_r1, hair_g1, hair_b1, hair_r2, hair_g2, hair_b2, hair_r3, hair_g3, hair_b3 = f2_lines[id+1].split()
                    patterns.append(f2_lines[id+1].split())
                    # patterns.append([skin_A, skin_B, eye_r, eye_g, eye_b, hair_r1, hair_g1, hair_b1])
                    
        count += 1

    if r:
        zipped_lists = list(zip(patterns, targets))
        random.shuffle(zipped_lists)
        patterns = np.array([x[0] for x in zipped_lists])
        targets = np.array([x[1] for x in zipped_lists])

    
    training_set = patterns[:int(.8*len(targets))]
    testing_set = patterns[int(.8*len(targets)):]
    training_targets = targets[:int(.8*len(targets))]
    testing_targets = targets[int(.8*len(targets)):]
    
    
    train_data = np.array(training_set)
    train_targets = np.array(training_targets)
    train_targets = to_categorical(train_targets)

    test_data = np.array(testing_set)
    test_targets = np.array(testing_targets)
    test_targets = to_categorical(test_targets)  

    train_data2 = []
    test_data2 = []
    for item in train_data:
        d = [eval(i) for i in item]
        train_data2.append(d)
    train_data = np.array(train_data2)
    for item in test_data:
        d = [eval(i) for i in item]
        test_data2.append(d)
    test_data = np.array(test_data2)

    return train_data, train_targets, test_data, test_targets


def nn():
    train_data, train_targets, test_data, test_targets = read_data()
    print(train_data[0])
    train_data, test_data = normalize_data(train_data), normalize_data(test_data)
    print(train_data[0])
    print(train_data.shape, train_targets.shape)
    print(test_data.shape, test_targets.shape)

    network = Sequential()
    network.add(Flatten(input_shape=(21,)))
    network.add(Dense(30, activation='relu', name='hidden', input_shape=(21,)))
    network.add(Dense(16, activation='relu', name='hidden2', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    network.add(Dense(4, activation='softmax', name='output'))
    network.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    # network.summary()
    print('pre_training_eval')
    network.evaluate(test_data,test_targets)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = network.fit(train_data, train_targets, epochs=250, batch_size=32, validation_split=.1, callbacks=[early_stopping])
    print('training_data_eval:')
    network.evaluate(train_data,train_targets)
    print('testing_data_eval')
    network.evaluate(test_data,test_targets)
    network.save('predict_season.h5')
    return history, network

def predict_image(image):
    data = ff.facial_features_and_values(image, True, True, 1)
    network = load_model('predict_season.h5')
    test_data = [
            data['skinLab'][0], data['skinLab'][1], data['skinLab'][2],
            data['eyeRGB'][0], data['eyeRGB'][1], data['eyeRGB'][2],
            data['eyeLab'][0], data['eyeLab'][1], data['eyeLab'][2],
            data['hairLab'][1], data['hairLab'][1], data['hairLab'][2],
            data['hairColors'][0][0], data['hairColors'][0][1], data['hairColors'][0][2],
            data['hairColors'][1][0], data['hairColors'][1][1], data['hairColors'][1][2],
            data['hairColors'][2][0], data['hairColors'][2][1], data['hairColors'][2][2]
    ]
    batch = np.array(test_data).reshape((1,21))
    output = network.predict(batch)
    return np.argmax(output)

#old
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
    file = open('patterns3.txt','w')
    count = 0
    file.write("skin_L skin_A skin_B eye_r eye_g eye_b eye_L eye_A eye_B hair_L hair_A hair_B hair_r1 hair_g1 hair_b1 hair_r2 hair_g2 hair_b2 hair_r3 hair_g3 hair_b3\n")
    for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
        data = ff.facial_features_and_values(image, False, True, 1)
        file.write("{:f} {:f} {:f} {:d} {:d} {:d} {:f} {:f} {:f} {:f} {:f} {:f} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} \n".format(
            data['skinLab'][0], data['skinLab'][1], data['skinLab'][2],
            data['eyeRGB'][0], data['eyeRGB'][1], data['eyeRGB'][2],
            data['eyeLab'][0], data['eyeLab'][1], data['eyeLab'][2],
            data['hairLab'][1], data['hairLab'][1], data['hairLab'][2],
            data['hairColors'][0][0], data['hairColors'][0][1], data['hairColors'][0][2],
            data['hairColors'][1][0], data['hairColors'][1][1], data['hairColors'][1][2],
            data['hairColors'][2][0], data['hairColors'][2][1], data['hairColors'][2][2]
        ))
        if count % 10 == 0:
            print(count) 
        # p = get_pattern(image)
        # for item in p:
        #     file.write(str(item) + " ")
        # file.write('\n')
        # if count == 2:
        #     break
        count += 1
    file.close()


def normalize_data(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data


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