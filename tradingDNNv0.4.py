# + Изучить временные ряды
# + проверить алгоритм деления по датам
# + Сделать первый прогноз
# + сделать интерпритацию результатов
# + повысить точность снизить потери
# + узывнать про байес и как с ним работать
# + убрать парраметр объема. оставлено

import pandas
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

period_of_chink = '30Y'
test_size = 0.2
epochs = 5
expected_accuracy = 0.99


class MyCallBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, log=None):
        if log is None:
            log = {}
        if log.get('acc') > expected_accuracy:
            print('\nReached ', expected_accuracy * 100, 'accuracy, so cancelling training!')
            self.model.stop_training = True


def get_data_normalization(dataframe):
    zscore = lambda x: (x - x.mean()) / x.std()

    dataframe_grouped = dataframe.groupby(by=['<TICKER>', pandas.Grouper(key='<DATE>', freq=period_of_chink)])

    df_normalized = dataframe_grouped[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].transform(zscore)

    return pandas.concat([dataframe[['<TICKER>', '<DATE>']], df_normalized], join='outer', axis=1)


def get_answers_dataframe(dataframe: object) -> object:
    def mask_first(x):
        result = np.ones_like(x)
        result[0] = 0
        return result

    def mask_last(x):
        result = np.ones_like(x)
        result[len(x) - 1] = 0
        return result

    dataframe_mask = dataframe[['<OPEN>']].transform(mask_last).astype(bool)
    answers_mask = dataframe[['<OPEN>']].transform(mask_first).astype(bool)

    dataframe_for_return = dataframe[dataframe_mask['<OPEN>']].reset_index(drop=True)
    answers_for_return = dataframe[answers_mask['<OPEN>']].reset_index(drop=True)

    return dataframe_for_return, answers_for_return


df = pandas.read_csv('total.txt',
                     delimiter=',',
                     parse_dates=['<DATE>'],
                     dtype={'<TICKER>': str,
                            '<OPEN>': float,
                            '<HIGH>': float,
                            '<LOW>': float,
                            '<CLOSE>': float,
                            '<VOL>': int},
                     decimal='.',
                     usecols=['<TICKER>', '<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']
                     )

df.index.name = 'id'

df = get_data_normalization(df)

df, df_ans = get_answers_dataframe(df)

# training_dataset = (
#     tf.data.Dataset.from_tensor_slices(
#         (
#             tf.cast(df.values, tf.float32),
#           # tf.cast(df_ans['<OPEN>'].values, tf.float32)
#         )
#     )
# )

training_dataset = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].values
training_dataset_ans = df_ans[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].values
# training_dataset_ans = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].values

X_train, X_test, Y_train, Y_test = train_test_split(
    training_dataset,
    training_dataset_ans,
    test_size=test_size,
    random_state=50,
    shuffle=True
)

callbacks = MyCallBack()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(5, activation='linear'),
])

model.compile(optimizer='Adagrad',
              loss='mean_squared_error',
              metrics=['accuracy']
              )

model.summary()

model.fit(X_train,
          Y_train,
          epochs=epochs,
          callbacks=[callbacks]
          )

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("\nFinally loss: %.2f, accuracy: %.2f" % (test_loss, test_acc))

df_for_pred = pandas.read_csv('ГАЗПРОМ ао [Price].txt',
                              delimiter=',',
                              parse_dates=['<DATE>'],
                              dtype={'<TICKER>': str,
                                     '<OPEN>': float,
                                     '<HIGH>': float,
                                     '<LOW>': float,
                                     '<CLOSE>': float,
                                     '<VOL>': int},
                              decimal='.',
                              usecols=['<TICKER>', '<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']
                              )

value_for_pred = df_for_pred[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].values[-1]

df_for_pred = get_data_normalization(df_for_pred)

value_for_pred_norm = df_for_pred[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']].values[-1]
value_for_pred_norm = value_for_pred_norm[None]

predict = model.predict(value_for_pred_norm)

print("Original value is:     %s\nPredict is:            %s" % (value_for_pred_norm, predict))

predicted_candle = (predict / value_for_pred_norm) * value_for_pred
predicted_candle = predicted_candle.T

print("\nForward candle is:\nOPEN:%.2f \nHIGH:%.2f \nLOW:%.2f \nCLOSE:%.2f \nVOL:%i" %
      (predicted_candle[0], predicted_candle[1], predicted_candle[2], predicted_candle[3], predicted_candle[4]))
