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
from imblearn.over_sampling import SMOTE

import facial_features as ff


def read_data(r=True):
    """
    read_data goes through the patterns and targets to assemble correct testing and training data.
    If r is true, the data will be shuffled before being assembled into the resepective data sets.
    """

    data = "targets.txt"
    patterns = []
    targets = []
    valid_targets = ["0", "1", "2", "3"]
    f = open(data, "r")
    count = 0
    f2 = open("patterns.txt", "r")
    f2_lines = f2.readlines()
    for line in f.readlines():
        if count == 0:
            pass
        else:
            l = line.split()
            if len(l) > 1:
                id = int(l[0])
                x = l[1]
                if x in valid_targets:
                    targets.append(int(x))
                    # get patterns
                    # avg_undertone, eye_color_r, eye_color_g, eye_color_b, l_eye, a_eye, b_eye, hair_color_r, hair_color_g, hair_color_b, l_hair, a_hair, b_hair = f2_lines[id].split()
                    (
                        skin_L,
                        skin_A,
                        skin_B,
                        eye_r,
                        eye_g,
                        eye_b,
                        eye_L,
                        eye_A,
                        eye_B,
                        hair_L,
                        hair_A,
                        hair_B,
                        hair_r1,
                        hair_g1,
                        hair_b1,
                        hair_r2,
                        hair_g2,
                        hair_b2,
                        hair_r3,
                        hair_g3,
                        hair_b3,
                    ) = f2_lines[id + 1].split()
                    # patterns.append(f2_lines[id+1].split())
                    patterns.append(
                        [
                            skin_A,
                            skin_B,
                            eye_L,
                            eye_r,
                            eye_g,
                            eye_b,
                            hair_L,
                            hair_r1,
                            hair_g1,
                            hair_b1,
                        ]
                    )

        count += 1

    if r:
        zipped_lists = list(zip(patterns, targets))
        random.shuffle(zipped_lists)
        patterns = np.array([x[0] for x in zipped_lists])
        targets = np.array([x[1] for x in zipped_lists])

    # training_set = patterns[: int(0.98 * len(targets))]
    # testing_set = patterns[int(0.98 * len(targets)) :]
    # training_targets = targets[: int(0.98 * len(targets))]
    # testing_targets = targets[int(0.98 * len(targets)) :]
    training_set = patterns
    # testing_set = patterns[int(0.98 * len(targets)) :]
    training_targets = targets
    # testing_targets = targets[int(0.98 * len(targets)) :]
    

    train_data = np.array(training_set)
    train_targets = np.array(training_targets)
    train_targets = to_categorical(train_targets)

    # test_data = np.array(testing_set)
    # test_targets = np.array(testing_targets)
    # test_targets = to_categorical(test_targets)
    test_targets = []
    test_data = []
    

    train_data2 = []
    test_data2 = []
    for item in train_data:
        d = [eval(i) for i in item]
        train_data2.append(d)
    train_data = np.array(train_data2)
    # for item in test_data:
    #     d = [eval(i) for i in item]
    #     test_data2.append(d)
    # test_data = np.array(test_data2)

    f.close()
    f2.close()
    return train_data, train_targets, test_data, test_targets


# fix to assume good data
def read_our_data():
    """
    read_our_data is similar to read_data but instead it parses the photos taken by us not in the database.
    It gets the patterns and targets so that they can be used for training and/or testing.
    """

    f2 = open("ourPatterns.txt", "r")
    f = open("ourTargets.txt", "r")
    count = 0
    f2_lines = f2.readlines()
    patterns = []
    targets = []
    valid_targets = ["0", "1", "2", "3"]
    for line in f.readlines():
        if count == 0:
            pass
        else:
            l = line.split()
            if len(l) > 1:
                id = int(l[0])
                x = l[1]
                if x in valid_targets:
                    targets.append(int(x))
                    # get patterns
                    # avg_undertone, eye_color_r, eye_color_g, eye_color_b, l_eye, a_eye, b_eye, hair_color_r, hair_color_g, hair_color_b, l_hair, a_hair, b_hair = f2_lines[id].split()
                    (
                        skin_L,
                        skin_A,
                        skin_B,
                        eye_r,
                        eye_g,
                        eye_b,
                        eye_L,
                        eye_A,
                        eye_B,
                        hair_L,
                        hair_A,
                        hair_B,
                        hair_r1,
                        hair_g1,
                        hair_b1,
                        hair_r2,
                        hair_g2,
                        hair_b2,
                        hair_r3,
                        hair_g3,
                        hair_b3,
                    ) = f2_lines[id + 1].split()
                    # patterns.append(f2_lines[id+1].split())
                    patterns.append(
                        [
                            skin_A,
                            skin_B,
                            eye_L,
                            eye_r,
                            eye_g,
                            eye_b,
                            hair_L,
                            hair_r1,
                            hair_g1,
                            hair_b1,
                        ]
                    )

        count += 1

    patterns = np.array(patterns, dtype=float)
    targets = np.array(targets)
    targets = to_categorical(targets)
    f2.close()
    f.close()
    return patterns, targets


def augment_data(patterns, targets):
    """
    augment_data takes in patterns and duplicates them with slight variations keeping the targets the same
    """

    new_patterns = []
    new_targets = []
    num_samples, num_features = patterns.shape
    shift_range = 3
    for i in range(len(patterns)):
        for j in range(3):  # >5
            np.random.seed(j+1)
            random_shift = np.random.uniform(-shift_range, shift_range, num_features)
            shifted_sample = patterns[i] + random_shift
            new_patterns.append(shifted_sample)
            new_targets.append(targets[i])
    return np.array(new_patterns), np.array(new_targets)


def get_training_data(predict_data=None):
    """
    2 purposes: first is to put together the full training and testing data.
    This includes augmenting the data, normalizing it and weighing it.
    The second is to correctly normalize any unssen patterns being run through the network to be predicted.
    """
    
    train_data, train_targets, test_data, test_targets = read_data()
    # our_patterns, our_targets = read_our_data() temp?
    test_data, test_targets = read_our_data()
 
    # our_patterns, our_targets = augment_data(our_patterns, our_targets)
    train_data, train_targets = augment_data(train_data, train_targets)
    # train_data = np.concatenate((train_data, our_patterns), axis=0)
    # train_targets = np.concatenate((train_targets, our_targets), axis=0)
    full_data = np.concatenate((train_data, test_data), axis=0)
    full_data_normal = normalize_data(full_data)

    train_data = full_data_normal[: len(train_data)]
    test_data = full_data_normal[len(train_data) :]

    smote = SMOTE(random_state=42)
    train_data, train_targets = smote.fit_resample(train_data, train_targets)
    # print(train_data.shape, train_targets.shape)
    feature_index = 0
    weight = 1.0  # 2 was best
    feature_index2 = 1
    weight2 = 1.0

    train_data[:, feature_index] *= weight
    train_data[:, feature_index2] *= weight2

    prepared_data_normal = []
    if predict_data is not None:
        # assume in batch state already
        # possibly make it so it can only be a batch of 1
        prepared_full_data = np.concatenate((predict_data, full_data), axis=0)
        prepared_full_data_normal = normalize_data(prepared_full_data)
        prepared_data_normal = prepared_full_data_normal[: len(predict_data)]
        prepared_data_normal[:, feature_index] *= weight
        prepared_data_normal[:, feature_index2] *= weight2

    return train_data, train_targets, test_data, test_targets, prepared_data_normal


def nn(e=150, file="predict_season.h5"):
    """
    Created and trains the neural network and then saves it.
    """
    
    train_data, train_targets, test_data, test_targets, _ = get_training_data()

    network = Sequential()
    network.add(Flatten(input_shape=(10,)))
    network.add(Dense(30, activation="relu", name="hidden", input_shape=(10,)))
    network.add(
        Dense(
            16,
            activation="relu",
            name="hidden2",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
    )
    network.add(Dense(4, activation="softmax", name="output"))
    network.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer="adam",
        metrics=["accuracy"],
    )
    # network.summary()
    # print("pre_training_eval")
    # network.evaluate(test_data, test_targets)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    print(train_data.shape, test_data.shape)
    
    history = network.fit(
        train_data,
        train_targets,
        epochs=e,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    print("db_eval:")
    db_data, db_targets = read_data()[0:2]
    db_data = get_training_data(db_data)[4]
    network.evaluate(db_data, db_targets)
    print("our_eval")
    network.evaluate(test_data, test_targets)
    network.save(file)
    return history, network


def load(file="predict_season.h5"):
    """
    Loads in a saves neural network.
    """
    
    return load_model(file)


def predict_image(image, data=None, file="best.h5"):
    """
    Takes in either an image path or image data and runs 
    it through the neural network to get a season prediction. 
    """
    
    network = load(file)
    if data == None:
        data = ff.facial_features_and_values(image, True, True, 1)

    test_data = [
        # data["skin_lab"][0],
        data["skin_lab"][1],
        data["skin_lab"][2],
        data["eye_lab"][0],
        data["eye_RGB"][0],
        data["eye_RGB"][1],
        data["eye_RGB"][2],
        # data["eye_lab"][1],
        # data["eye_lab"][2],
        data["hair_lab"][0],
        # data["hair_lab"][1],
        # data["hair_lab"][2],
        data["hair_colors"][0][0],
        data["hair_colors"][0][1],
        data["hair_colors"][0][2],
        # data["hair_colors"][1][0],
        # data["hair_colors"][1][1],
        # data["hair_colors"][1][2],
        # data["hair_colors"][2][0],
        # data["hair_colors"][2][1],
        # data["hair_colors"][2][2],
    ]
    # print(test_data)
    test_data = np.array(test_data).reshape((1, 10))
    test_data = get_training_data(test_data)[4]
    # print(test_data)
    output = network.predict(test_data)
    return np.argmax(output)

def write_pattern(ours=False, filename="patterns.txt"):
    """
    Iterates through all images getting their facial data and writes it neatly as a pattern in to a file.
    """
    
    file = open(filename, "w")
    count = 0
    file.write(
        "skin_L skin_A skin_B eye_r eye_g eye_b eye_L eye_A eye_B hair_L hair_A hair_B hair_r1 hair_g1 hair_b1 hair_r2 hair_g2 hair_b2 hair_r3 hair_g3 hair_b3\n"
    )
    if ours:
        folder = "./OurPhotos/*.jpg"
    else:
        folder = "./ChicagoFaceDatabaseImages/*.jpg"
    for image in glob.glob(folder):
        data = ff.facial_features_and_values(image, ours, True, 1)
        file.write(
            "{:f} {:f} {:f} {:d} {:d} {:d} {:f} {:f} {:f} {:f} {:f} {:f} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} \n".format(
                data["skin_lab"][0],
                data["skin_lab"][1],
                data["skin_lab"][2],
                data["eye_RGB"][0],
                data["eye_RGB"][1],
                data["eye_RGB"][2],
                data["eye_lab"][0],
                data["eye_lab"][1],
                data["eye_lab"][2],
                data["hair_lab"][1],
                data["hair_lab"][1],
                data["hair_lab"][2],
                data["hair_colors"][0][0],
                data["hair_colors"][0][1],
                data["hair_colors"][0][2],
                data["hair_colors"][1][0],
                data["hair_colors"][1][1],
                data["hair_colors"][1][2],
                data["hair_colors"][2][0],
                data["hair_colors"][2][1],
                data["hair_colors"][2][2],
            )
        )
        if count % 10 == 0:
            print(count)
        count += 1
    file.close()


def normalize_data(data):
    """
    Normalizes an array of data based on the largest and smallest numbers in each feature.
    """
    
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data


#can be incorporated as just being the test data when the neural network is trained.
def test_all(file="best.h5"):
    """
    Iterates through a folder of images to see how well a neural network did.
    """
    
    model = load(file)
    patterns, targets = read_our_data()
    # patterns = np.array(patterns).reshape((1, 10))
    patterns = get_training_data(patterns)[4]
    outputs = model.predict(patterns)
    correct = 0
    for i in range(len(outputs)):
        network_answer = np.argmax(outputs[i])
        correct_answer = np.argmax(targets[i])
        if network_answer == correct_answer:
            correct += 1
            print(str(network_answer) + " is correct (photo " + str(i) + ")")
    success = correct / len(outputs)
    return outputs, success


def run(epochs=150, file="network2.h5"):
    h, n = nn(epochs, file)
    plot_history(h)
    o, s = test_all(file)
    return s


def plot_history(history):
    loss_values = history.history["loss"]
    accuracy_values = history.history["accuracy"]
    validation = "val_loss" in history.history
    if validation:
        val_loss_values = history.history["val_loss"]
        val_accuracy_values = history.history["val_accuracy"]
    epoch_nums = range(1, len(loss_values) + 1)
    plt.figure(figsize=(12, 4))  # width, height in inches
    plt.subplot(1, 2, 1)
    if validation:
        plt.plot(epoch_nums, loss_values, "r", label="Training loss")
        plt.plot(epoch_nums, val_loss_values, "r--", label="Validation loss")
        plt.title("Training/validation loss")
        plt.legend()
    else:
        plt.plot(epoch_nums, loss_values, "r", label="Training loss")
        plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    if validation:
        plt.plot(epoch_nums, accuracy_values, "b", label="Training accuracy")
        plt.plot(epoch_nums, val_accuracy_values, "b--", label="Validation accuracy")
        plt.title("Training/validation accuracy")
        plt.legend()
    else:
        plt.plot(epoch_nums, accuracy_values, "b", label="Training accuracy")
        plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()
