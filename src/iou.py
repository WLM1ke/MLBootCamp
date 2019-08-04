"""Реализация для Keras метрики intersection over union."""
import tensorflow as tf


def boxes_area(boxes: tf.Tensor) -> tf.Tensor:
    """Площадь рамок."""
    zero = tf.constant(0, dtype=boxes.dtype)
    width = tf.maximum(boxes[:, 2] - boxes[:, 0], zero)
    height = tf.maximum(boxes[:, 3] - boxes[:, 1], zero)
    return width * height


def intersection_area(boxes_true: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """Область пересечения наборов рамок."""
    x_min = tf.stack([boxes[:, 0], boxes_true[:, 0]], axis=-1)
    y_min = tf.stack([boxes[:, 1], boxes_true[:, 1]], axis=-1)
    x_max = tf.stack([boxes[:, 2], boxes_true[:, 2]], axis=-1)
    y_max = tf.stack([boxes[:, 3], boxes_true[:, 3]], axis=-1)

    x_min = tf.reduce_max(x_min, axis=-1)
    y_min = tf.reduce_max(y_min, axis=-1)
    x_max = tf.reduce_min(x_max, axis=-1)
    y_max = tf.reduce_min(y_max, axis=-1)

    boxes_inter = tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    return boxes_area(boxes_inter)


def intersection_over_union(boxes_true: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """Метрика intersection over union для прямоугольных рамок.

    Выдается отрицательное значения для целей минимизации функции потерь в Keras.

    :param boxes: Тензор батча формы (None, 4) содержащий прогнозные прямоугольные рамки в формате
        (x_min, y_min, x_max, y_max).
    :param boxes_true: Тензор батча формы (None, 4) содержащий фактические прямоугольные рамки в формате
        (x_min, y_min, x_max, y_max).
    :return:
        Значение средней метрики intersection over union для батча со знаком минус.
    """
    inter_area = intersection_area(boxes_true, boxes)
    area = boxes_area(boxes)
    area_true = boxes_area(boxes_true)
    iou = inter_area / (area + area_true - inter_area + tf.keras.backend.epsilon())
    return -tf.reduce_mean(iou)
