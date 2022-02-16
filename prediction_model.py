from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

predict_len = 7


def dict_fetchall(cursor):
    columns = [col[0] for col in cursor.description]
    ans = []
    for row in cursor.fetchall():
        m = dict()
        for i in range(len(row)):
            m[columns[i]] = str(row[i])
        ans.append(m)
    return ans


factor_names = ['temprature', 'ph', 'do', 'conductivity', 'permanganate', 'nh3n', 'tp', 'tn', 'turbidity', ]
scalars = dict({
    "temprature": 100,
    "ph": 10,
    "do": 100,
    "conductivity": 1000,
    "permanganate": 100,
    "turbidity": 1000,
    "nh3n": 10,
    "tp": 10,
    "tn": 100,
    "COD_V": 100,
})
char2index = dict((name, i) for i, name in enumerate(factor_names))
display_factor_names = ['nh3n', 'tp', 'permanganate']

raw_data = pd.read_excel("data/lianghekou.xlsx")
raw_data = raw_data.sort_values(by="monitoredtime")[factor_names]
raw_data = raw_data.dropna(axis=0, how='any')
for i in factor_names:
    raw_data[i] = raw_data[i] / scalars[i]

train_x, train_y = [], []

pre_i = 0
for i in range(len(raw_data)):
    if i + 2 * predict_len < len(raw_data):
        it = raw_data[i: i + predict_len][factor_names]
        nit = raw_data[i + predict_len: i + 2 * predict_len][display_factor_names]
        train_x.append(it.values.tolist())
        train_y.append(nit.values.flatten().tolist())
    else:
        break

all_x, all_y = abs(np.array(train_x, dtype='float32')), abs(np.array(train_y, dtype='float32'))
train_x, train_y = all_x[:int(len(all_x) * 0.9)], all_y[:int(len(all_x) * 0.9)]
test_x, test_y = all_x[int(len(all_x) * 0.9):], all_y[int(len(all_x) * 0.9):]

model = Sequential()
model.add(LSTM(128,  activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(len(display_factor_names) * predict_len))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, amsgrad=True), metrics=['mse'])
epchos = 10
history = model.fit(train_x, train_y, epochs=epchos, batch_size=predict_len, validation_data=(test_x, test_y), verbose=1, )
model.save("data/model.h5")

with open('data/history.txt', 'wb') as ft:
    pickle.dump(history.history, ft)

# model = load_model("data/model.h5")

plt.figure(1)
plt.plot(range(epchos), history.history['mse'])
plt.title('mse')

test_predict_y = model.predict(test_x)

plt.figure(2)

real = test_y.reshape((test_y.shape[0], predict_len, len(display_factor_names)))[0]
prediction = test_predict_y.reshape((test_predict_y.shape[0], predict_len, len(display_factor_names)))[0]
for i, factor_name in enumerate(display_factor_names):
    plt.subplot(1, len(display_factor_names), i + 1)

    if factor_name == 'permanganate':
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        ax = plt.gca()
        ax.set_ylim(0, 20)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    elif factor_name == 'nh3n':
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    else:
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.plot(range(predict_len), [j[i] * scalars[factor_name] for j in prediction],
             label=f"predict",
             color="red")
    plt.plot(range(predict_len), [j[i] * scalars[factor_name] for j in real],
             label=f"real")
    plt.legend()
    plt.title(f"{factor_name}")

plt.show()
