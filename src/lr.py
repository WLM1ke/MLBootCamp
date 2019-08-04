"""Callback(и) для Keras тестирующие и управляющие learning rate."""
from typing import Generator
from typing import Union

import keras
from keras import callbacks
from keras import backend
import pandas as pd
import matplotlib.pyplot as plt



class MaxLRTest(callbacks.Callback):
    """Тест на максимальный learning rate.

    Learning rate экспоненциально повышается от начального до максимального значения за определенное число итераций.
    Собирается статистика loss и выбирается минимальное скользящее среднее.
    """
    def __init__(self, lr_base=1.0e-10, lr_max=1.0, steps=10000):
        super().__init__()

        self.lr_base = lr_base
        self.lr_mul = (lr_max / lr_base) ** (1 / steps)
        self.steps = steps

        self.iterations = None
        self.history = None

    def on_train_begin(self, logs=None):
        """Устанавливает начальную скорость обучения и удаляет историю."""
        self.iterations = 0
        self.history = {}
        backend.set_value(self.model.optimizer.lr, self.lr_base)

    def on_batch_end(self, batch, logs=None):
        """Сохраняет историю, изменяет learning rate и останавливает обучение."""
        self.history.setdefault("iterations", []).append(self.iterations)
        self.history.setdefault("loss", []).append(logs["loss"])

        lr = backend.get_value(self.model.optimizer.lr)
        self.history.setdefault("lr", []).append(lr)

        backend.set_value(self.model.optimizer.lr, lr * self.lr_mul)

        self.iterations += 1
        if self.iterations == self.steps:
            self.model.stop_training = True


def get_max_lr(
        model: keras.Model,
        generator: Union[keras.utils.Sequence, Generator],
        lr_base: float = 1.0e-10,
        lr_max: float = 1.0,
        steps: int = 10000,
        smooth: int = 1000,
        plot: bool = True
) -> float:
    """Осуществляет тестирование модели на максимальный learning rate и при необходимости рисует график

    :param model:
        Скомпилированная Keras модель.
    :param generator:
        Генератор обучающих примеров.
    :param lr_base:
        Начальная скорость обучения.
    :param lr_max:
        Максимальная скорость обучения.
    :param steps:
        Количество промежуточных шагов.
    :param smooth:
        Количество шагов для сглаживания.
    :param plot:
        Нужно ли рисовать график.
    :return:
        Максимальная скорость обучения
    """
    test = MaxLRTest(lr_base, lr_max, steps)
    model.fit_generator(
        generator,
        steps_per_epoch=steps,
        epochs=1,
        callbacks=[test]
    )
    history = pd.DataFrame(test.history).set_index("lr").loss.rolling(smooth).mean()
    lr = history.idxmin()
    print(f"Max speed learning rate  - {lr:.1e}")
    if plot:
        history.plot(logx=True, figsize=(16, 8))
        plt.show()
    return lr


class DecayingLR(callbacks.Callback):
    """Схема обучения на основе повышения и снижения learning rate.

    Сначала для разогрева learning rate повышается от нуля до максимального значения, а потом начинается снижение по
    мере того, как loss перестает снижаться. Скорость снижения loss отлеживается на каждой итерации в виде
    экспоненциального скользящего среднего. Снижение продолжается пока val_loss не будет меняться сильнее небольшой
    величины в течении установленного числа периодов.
    """

    def __init__(
            self,
            lr_max: float,
            warm_up: int = 1,
            decay_per_epoch: float = 0.5,
            wait: int = 3,
            epsilon: float = 0.0001,
            verbose: bool = True
    ):
        """Схема обучения на основе повышения и снижения learning rate.

        Сначала для разогрева learning rate повышается от нуля до максимального значения, а потом начинается снижение по
        мере того, как loss перестает снижаться. Скорость снижения loss отлеживается на каждой итерации в виде
        экспоненциального скользящего среднего. Снижение продолжается пока val_loss не будет меняться сильнее небольшой
        величины в течении установленного числа периодов.

        :param lr_max:
            Максимальный learning rate.
        :param warm_up:
            Период разогрева в эпохах.
        :param decay_per_epoch:
            На сколько снижать learning rate за эпоху, при остановке снижения loss.
        :param wait:
            В течении скольких периодов val_loss должно мало меняться для остановки обучения.
        :param epsilon:
            Минимальное изменение val_loss для остановки.
        :param verbose:
            Нужно ли печатать данные по скорости обучения и предстоящей остановке.
        """
        super().__init__()

        self.lr_max = lr_max
        self.warm_up = warm_up
        self.decay_per_epoch = decay_per_epoch
        self.epsilon = epsilon
        self.wait = wait
        self.verbose = verbose

        self.iterations = None
        self.history = None

        self.d_loss_iterations = None
        self.d_loss = None
        self.d_iterations = None
        self.d_iterations2 = None
        self.d_1 = None

        self.steps = None
        self.decay = None

        self.wait_count = None
        self.prev_loss = None

    def on_train_begin(self, logs=None):
        """Сброс всех параметров."""
        self.iterations = 0
        self.history = {}

        self.d_loss_iterations = 0.0
        self.d_loss = 0.0
        self.d_iterations = 0.0
        self.d_iterations2 = 0.0
        self.d_1 = 0.0

        self.steps = self.params["steps"]
        self.decay = self.decay_per_epoch ** (1 / self.steps)

        self.wait_count = 0
        self.prev_loss = None

        backend.set_value(self.model.optimizer.lr, self.lr())

    def make_decay(self, old, new):
        """Обновляет значение с учетом фактора затухания."""
        decay = self.decay
        return old * decay + new * (1 - decay)

    def lr(self):
        """Рассчитывает новый learning rate."""
        warm_up = (self.iterations + 1) / (self.steps * self.warm_up)
        if warm_up <= 1:
            return self.lr_max * warm_up

        lr = self.history["lr"][-1]
        speed = self.history["speed"][-1]
        if speed > 0:
            return lr * self.decay
        return lr

    def on_batch_end(self, batch, logs=None):
        """Сохраняет статистику и обновляет скорость learning rate."""
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.history.setdefault("iterations", []).append(self.iterations)
        self.history.setdefault("lr", []).append(backend.get_value(self.model.optimizer.lr))

        self.d_loss_iterations = self.make_decay(self.d_loss_iterations, logs["loss"] * self.iterations)
        self.d_loss = self.make_decay(self.d_loss, logs["loss"])
        self.d_iterations = self.make_decay(self.d_iterations, self.iterations)
        self.d_iterations2 = self.make_decay(self.d_iterations2, self.iterations ** 2)
        self.d_1 = self.make_decay(self.d_1, 1)

        cov = self.d_loss_iterations - self.d_loss * self.d_iterations / self.d_1
        var = self.d_iterations2 - self.d_iterations ** 2 / self.d_1

        speed = 0
        if var != 0:
            speed = self.steps * cov / var
        self.history.setdefault("speed", []).append(speed)

        self.iterations += 1
        backend.set_value(self.model.optimizer.lr, self.lr())

    def on_epoch_end(self, epoch, logs=None):
        """Обработка остановки и распечатка данных о скорости обучения и остановке."""
        if self.prev_loss is None:
            self.prev_loss = logs["val_loss"]
        elif abs(self.prev_loss - logs["val_loss"]) < self.epsilon:
            self.wait_count += 1
        else:
            self.wait_count = 0
        self.prev_loss = logs["val_loss"]

        if self.wait == self.wait_count:
            self.model.stop_training = True

        if self.verbose:
            lr = self.history["lr"][-1]
            speed = self.history["speed"][-1]
            print(f"Learning rate: {lr:.1e}")
            print(f"Speed per epoch: {speed:.4f}")
            print(f"Wait to stop: {self.wait - self.wait_count}\n")
