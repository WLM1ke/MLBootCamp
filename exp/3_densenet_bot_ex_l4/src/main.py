"""Модель на основе Keras с сырыми данными а-ля RezNet с бутылочным блоком."""
import pathlib

from keras import backend
from keras import callbacks
from keras import layers
from keras import models
import numpy as np
import pandas as pd
from keras import optimizers
from sklearn import model_selection

from src import conv1d
from src import load_data
from src import iou
from src import lr

COORDINATES = load_data.COORDINATES

DATA_PATH = pathlib.Path("../processed")
DATA_PATH.mkdir(parents=True, exist_ok=True)

BLOCKS = 4
LINK = conv1d.Links.DENSENET
CHANNELS = 64
LAYERS_TYPE = conv1d.Layers.BOTTLENECK
SE = True


FOLDS = 10
EPOCHS = 100
LR_MAX = 8.3e-04


def yield_batch(data):
    """Обучающие примеры."""
    votes, answers = data
    item_ids = list(set(votes.index))
    while True:
        item_id = np.random.choice(item_ids, 1)
        forecasts = votes.loc[item_id].set_index("userId")
        x = np.zeros((1, len(forecasts), 4),)
        y = np.zeros((1, 4))
        x[0] = forecasts.sample(len(forecasts))
        y[0] = answers.loc[item_id]
        yield x, y


def yield_batch_val(data):
    """Примеры для валидации."""
    votes, answers = data
    item_ids = set(votes.index)
    while True:
        for item_id in item_ids:
            forecasts = votes.loc[item_id].set_index("userId")
            x = np.zeros((1, len(forecasts), 4),)
            y = np.zeros((1, 4))
            x[0] = forecasts
            y[0] = answers.loc[item_id]
            yield x, y


def yield_batch_test(data):
    """Примеры для тестирования."""
    item_ids = data.index.unique()
    for item_id in item_ids:
        forecasts = data.loc[item_id].set_index("userId")
        x = np.zeros((1, len(forecasts), 4),)
        x[0] = forecasts
        yield x


def make_model(filters=CHANNELS):
    """Создает сеть на основе сгенеренных признаков."""
    backend.clear_session()

    y = x = layers.Input(shape=(None, 4))
    y = conv1d.make_net(y, BLOCKS, LINK, CHANNELS, LAYERS_TYPE, se=SE)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dense(
        units=filters // 2,
        activation="relu"
    )(y)
    y = layers.Dense(
        units=4,
        activation=None
    )(y)

    model = models.Model(inputs=x, outputs=y)
    model.summary()
    return model


def train_model(data_train, data_val, lr_max=LR_MAX, epochs=EPOCHS):
    """Обучение модели."""
    steps_per_epoch = 1000
    path = str(DATA_PATH / "model.h5")

    model = make_model()
    model.compile(
        optimizer=optimizers.Nadam(),
        loss="mae",
        metrics=[iou.intersection_over_union],
    )
    model.fit_generator(
        yield_batch(data_train),
        steps_per_epoch=3000,
        epochs=1,
        validation_data=yield_batch_val(data_val),
        validation_steps=len(data_val[1].index),
    )

    model.compile(
        optimizer=optimizers.Nadam(),
        loss=iou.intersection_over_union
    )
    lr_callback = lr.DecayingLR(lr_max=lr_max)
    save_callback = callbacks.ModelCheckpoint(
        path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    )
    cb = [save_callback, lr_callback]
    rez = model.fit_generator(
        yield_batch(data_train),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=cb,
        validation_data=yield_batch_val(data_val),
        validation_steps=len(data_val[1].index),
    )

    model = models.load_model(
        path,
        custom_objects={"intersection_over_union": iou.intersection_over_union}
    )

    return rez, model


def train_oof(train_set, test_set):
    """Обучение OOF."""
    x_train, y_train = train_set
    x_test = test_set

    oof = pd.DataFrame(0, index=y_train.index, columns=COORDINATES)
    prediction = pd.DataFrame(0, index=x_test.index.unique(), columns=COORDINATES)
    scores = []
    folds = model_selection.KFold(n_splits=FOLDS, shuffle=True)

    for n, (index_train, index_valid) in enumerate(folds.split(y_train), 1):
        print(f"\nFold - {n} / {FOLDS}")

        ids_train = y_train.index[index_train]
        ids_valid = y_train.index[index_valid]

        data_train = (x_train.loc[ids_train], y_train.loc[ids_train])
        data_val = (x_train.loc[ids_valid], y_train.loc[ids_valid])

        rez, model = train_model(data_train, data_val)
        scores.append(min(rez.history["val_loss"]))

        feat = yield_batch_test(data_val[0])
        df = model.predict_generator(feat, steps=len(data_val[0].index.unique()))
        oof.loc[ids_valid] = df

        feat = yield_batch_test(x_test)
        df = model.predict_generator(feat, steps=len(x_test.index.unique()))
        prediction += df / FOLDS

    mean = np.mean(scores)
    err = np.std(scores) / len(scores) ** 0.5
    print(f"IOU на кроссвалидации: " + str(-np.round(sorted(scores), 5)))
    print(f"IOU среднее: {-mean:0.5f} +/- {2 * err:0.5f}")

    oof.to_csv(DATA_PATH / f"oof-{-mean - 2 * err:0.5f}.csv", header=False)
    prediction.to_csv(DATA_PATH / f"sub-{-mean - 2 * err:0.5f}.csv", header=False)


def train():
    """Тренировка модели."""
    train_set = (load_data.train_x(), load_data.train_y())
    test_set = load_data.test_x()
    train_oof(train_set, test_set)


def find_lr():
    """Поиск максимального lr."""
    train_set = (load_data.train_x(), load_data.train_y())
    model = make_model()
    model.compile(
        optimizer=optimizers.Nadam(),
        loss="mae",
        metrics=[iou.intersection_over_union],
    )
    model.fit_generator(
        yield_batch(train_set),
        steps_per_epoch=30,
        epochs=1
    )
    model.compile(
        optimizer=optimizers.Nadam(),
        loss=iou.intersection_over_union
    )

    lr.get_max_lr(model, yield_batch(train_set))


if __name__ == '__main__':
    train()
