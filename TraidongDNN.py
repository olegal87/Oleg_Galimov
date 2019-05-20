import csv
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

expected_accuracy = 0.998

class myCallBack(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, log={}):
       if(log.get('acc')>expected_accuracy):
          print('\nReached ',expected_accuracy*100,'accuracy, so cancelling training!' )
          self.model.stop_training = True

def importdata(reader_object):
    dataset = []

    for idx, varname in enumerate(reader_object):
        if idx == 0:
            varnames = varname
        else:
            dataset.append(varname)

    return varnames, dataset

def get_data_normalization(data):
    vol_mean = np.mean(data)
    vol_std = np.std(data)
    # normalization
    data = [
        (cur_vol - vol_mean) / vol_std
        for cur_vol in data
    ]
    return data

def push_data_into_var(dataset, varname):
    data = [
        float(voldata[var_names.index(varname)])
        for voldata in dataset
    ]
    return data

def convert_to_tensor(arg):

    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg


with open('Сбербанк [Price].txt', 'rt') as f:
    reader = csv.reader(f, delimiter=',', skipinitialspace=True, )

    [var_names, dataset] = importdata(reader)

    # removing characters from parrametr names
    chars_for_remove = "\<|\>"
    var_names = [
        re.sub(chars_for_remove, '', var)
        for var in var_names
    ]

    # lower case
    var_names = [var.lower() for var in var_names]

    #  push rates volume data into variable
    open_price = push_data_into_var(dataset, "open")
    close = push_data_into_var(dataset, "close")
    high = push_data_into_var(dataset, "high")
    low = push_data_into_var(dataset, "low")
    vol = push_data_into_var(dataset, "vol")

    open_price = get_data_normalization(open_price)
    close = get_data_normalization(close)
    high = get_data_normalization(high)
    low = get_data_normalization(low)
    vol = get_data_normalization(vol)

    #create dataset for nn
    length = len(open_price)

    dataset = [
        open_price,
        close,
        high,
        low,
        vol
    ]
    for idx, var in enumerate(dataset):
        res = var.copy()
        del res[length-1]
        dataset[idx] = res

    #create answers for nn

    answers = [
        open_price,
        close,
        high,
        low,
        vol
    ]
    for idx, var in enumerate(answers):
        del var[0]
        answers[idx] = var

    #separate datasets to training set and test set
    dataset = np.array(dataset).transpose()
    answers = np.array(answers).transpose()

    dataset = tf.data.Dataset.from_tensor_slices((dataset, answers))


    #dataset = convert_to_tensor(dataset)
    print(dataset)

  #  answers = convert_to_tensor(answers)

    X_train, X_test, y_train, y_test = train_test_split(
            dataset, answers, test_size = 0.2, random_state = 38)

    callbacks = myCallBack()

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,), batch_size=5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, callbacks=[callbacks],batch_size=5)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(test_acc)





