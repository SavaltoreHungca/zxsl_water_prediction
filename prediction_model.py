from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

predict_len = 120
input_size = 240
epchos = 10

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
    if i + input_size + predict_len < len(raw_data):
        it = raw_data[i: i + input_size][factor_names]
        nit = raw_data[i + input_size: i + input_size + predict_len][display_factor_names]
        train_x.append(it.values.tolist())
        train_y.append(nit.values.flatten().tolist())
    else:
        break

all_x, all_y = abs(np.array(train_x, dtype='float32')), abs(np.array(train_y, dtype='float32'))
train_x, train_y = all_x[:int(len(all_x) * 0.9)], all_y[:int(len(all_y) * 0.9)]
t_x, t_y = all_x[int(len(all_x) * 0.9):], all_y[int(len(all_y) * 0.9):]
val_x, val_y = t_x[:int(len(t_x) * 0.6)], t_y[:int(len(t_y) * 0.6)]
test_x, test_y = t_x[int(len(t_x) * 0.6):], t_y[int(len(t_y) * 0.6):]


model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(len(display_factor_names) * predict_len))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, amsgrad=True), metrics=['mse'])
history = model.fit(train_x, train_y, epochs=epchos, batch_size=predict_len, validation_data=(test_x, test_y),
                    verbose=1, ).history
model.save("data/model.h5")

with open('data/history.txt', 'wb') as ft:
    pickle.dump(history, ft)


# model = load_model("data/model.h5")
# with open('data/history.txt', 'rb') as ft:
#     history = pickle.load(ft)

plt.figure(1)
plt.plot(range(epchos), history['mse'])
plt.title('mse')

test_predict_y = model.predict(val_x)

plt.figure(2)

real = val_y.reshape((val_y.shape[0], predict_len, len(display_factor_names)))
prediction = test_predict_y.reshape((test_predict_y.shape[0], predict_len, len(display_factor_names)))
for i, factor_name in enumerate(display_factor_names):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    plt.subplot(1, len(display_factor_names), i + 1)
    ax = plt.gca()
    if factor_name == 'permanganate':
        ax.set_ylim(0, 20)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    elif factor_name == 'nh3n':
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    else:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.plot(range(predict_len), [j[i] * scalars[factor_name] for j in prediction[0]],
             label=f"predict",
             color="red")
    plt.plot(range(predict_len), [j[i] * scalars[factor_name] for j in real[0]],
             label=f"real")
    plt.legend()
    plt.title(f"{factor_name}")

plt.figure(3)
for i, factor_name in enumerate(display_factor_names):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    plt.subplot(len(display_factor_names), 1, i + 1)
    ax = plt.gca()
    if factor_name == 'permanganate':
        ax.set_ylim(0, 20)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    elif factor_name == 'nh3n':
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    else:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.plot(range(len(prediction)), [j[0][i] * scalars[factor_name] for j in prediction],
             label=f"predict",
             color="red")
    plt.plot(range(len(real)), [j[0][i] * scalars[factor_name] for j in real],
             label=f"real")
    plt.legend()
    plt.title(f"{factor_name}")

plt.show()
