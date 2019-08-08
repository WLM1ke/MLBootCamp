"""Модель на основе Keras с сгенеренными признаками."""
import pathlib

from keras import backend
from keras import callbacks
from keras import layers
from keras import models
import numpy as np
import pandas as pd
from keras import optimizers
from sklearn import model_selection

from src import load_data
from src import iou
from src import lr

COORDINATES = load_data.COORDINATES

DATA_PATH = pathlib.Path("../processed")
DATA_PATH.mkdir(parents=True, exist_ok=True)

FOLDS = 10
EPOCHS = 100
BATCH = 8
UNITS = 16
HEIGHT = 4
LR_MAX = 1.1e-03


def make_features(votes):
    """Создает признаки для обучения."""
    feat = votes.groupby("itemId")[COORDINATES].median()
    feat2 = votes.groupby("itemId")[COORDINATES].agg(["min", "max", "std", "mean"])
    feat3 = votes[["userId"]].groupby("itemId").count()
    return pd.concat([feat, feat2, feat3], axis=1)


def yield_batch(data, batch=BATCH):
    """Обучающие примеры."""
    votes, answers = data
    feat = make_features(votes)
    item_ids = votes.index.unique()
    while True:
        item_id = np.random.choice(item_ids, batch)
        x = feat.loc[item_id]
        y = answers.loc[item_id]
        yield x, y


def yield_batch_val(data):
    """Примеры для валидации."""
    votes, answers = data
    feat = make_features(votes)
    item_ids = votes.index.unique()
    while True:
        x = feat.loc[item_ids]
        y = answers.loc[item_ids]
        yield x, y


def yield_batch_test(data):
    """Примеры для тестирования."""
    feat = make_features(data)
    item_ids = data.index.unique()
    for item_id in item_ids:
        x = feat.loc[[item_id]]
        yield x


def make_model(units=UNITS, height=HEIGHT):
    """Создает сеть на основе сгенеренных признаков."""
    backend.clear_session()

    y = x = layers.Input(shape=(21,))

    for i in range(height, 0, -1):
        y = layers.Dense(
            units=units * 2 ** (i - 1),
            activation="relu"
        )(y)

    y = layers.Dense(
        units=4,
        activation=None
    )(y)

    model = models.Model(inputs=x, outputs=y)
    model.summary()
    return model


def train_model(data_train, data_val, lr_max=LR_MAX, batch=BATCH, epochs=EPOCHS):
    """Обучение модели."""
    steps_per_epoch = 1000 // batch
    path = str(DATA_PATH / "model.h5")

    model = make_model()
    model.compile(
        optimizer=optimizers.Nadam(),
        loss="mae",
        metrics=[iou.intersection_over_union],
    )
    model.fit_generator(
        yield_batch(data_train, batch),
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        validation_data=yield_batch_val(data_val),
        validation_steps=1,
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
        yield_batch(data_train, batch),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=cb,
        validation_data=yield_batch_val(data_val),
        validation_steps=1,
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


if __name__ == '__main__':
    train_set = (load_data.train_x(), load_data.train_y())
    test_set = load_data.test_x()
    train_oof(train_set, test_set)
