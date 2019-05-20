import tensorflow as tf
from pathlib import Path

expected_accuracy = 0.998

class myCallBack(tf.keras.callbacks.Callback):

   def on_epoch_end(self, epoch, log={}):

       if(log.get('acc')>expected_accuracy):
          print('\nReached ',expected_accuracy*100,'accuracy, so cancelling training!' )
          self.model.stop_training = True


def create_answerfile(file_name):

    path = str(Path(__file__).resolve().parent)

    file_name = path +"\\"+ str(file_name)

    original = file_name
    answers = file_name + "_answers"

    # delete first line from answers (not header)
    with open(original, 'r') as f:
        with open(answers, 'w') as f1:

            for i, line in enumerate(f):
                if i == 0:
                    line = line.replace(">",">_answer")
                if i == 1:
                    continue

                f1.write(line)

    return answers


# def input_fn(dataset):
#
        # feature_dict = {
        #     'open_',
        #
        # }
#     open_ = tf.feature_column.numeric_column('<OPEN>')
#     high_ = tf.feature_column.numeric_column('<HIGH>')
#     low_ = tf.feature_column.numeric_column('<LOW>')
#     close_ = tf.feature_column.numeric_column('<CLOSE>')
#     volume= tf.feature_column.numeric_column('<VOL>')
#
#    ...  # manipulate dataset, extracting the feature dict and the label
#    return feature_dict, label

# Creates a dataset that reads all of the records from two CSV files, each with
# five float columns

print("Loading data")

tf.enable_eager_execution()

data_filename = "Сбербанк [Price].txt"

record_defaults = [tf.float32] * 5 # five required float columns

dataset = tf.data.experimental.CsvDataset(
    data_filename,
    record_defaults,
    header=True,
    select_cols=[4, 5, 6, 7, 8]
)



ans_dataset_file = create_answerfile(data_filename)

#Push answers file into dataset

ans_dataset = tf.data.experimental.CsvDataset(
    ans_dataset_file,
    record_defaults,
    header=True,
    select_cols=[4, 5, 6, 7, 8],
)

#should be considered delete answers dataset file cause its loaded to memory

dataset = dataset, ans_dataset

# create itirator for dataset

#iterator = dataset.make_one_shot_iterator()


# print(dataset)
callbacks = myCallBack()

#declare a model

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    dataset,
    shuffle=False,
    epochs=20,
    callbacks=[callbacks],
    batch_size=5,
    steps_per_epoch=100
)
#test_loss, test_acc = model.evaluate(X_test, y_test)


#print(list(dataset)[-1])





#
#
# data, answers = create_datasetfiles(data_filename)
# print(data)



# answers = copy.copy(dataset)
# print(len(list(answers)))
# list(answers).pop(1)
#
# print(len(list(answers)))


# for data in answers:
#     print(list(data))

# print("Creating iritator")

# feature_dict, label = input_fn(dataset)