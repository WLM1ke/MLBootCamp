"""Загрузчики данных."""
import pathlib

import pandas as pd

COORDINATES = ["Xmin", "Ymin", "Xmax", "Ymax"]
PATH = pathlib.Path(__file__).parents[1] / "raw"


def train_x():
    """Загрузка обучающих прогнозов."""
    return pd.read_csv(PATH / "train_data.csv").set_index("itemId")


def train_y():
    """Загрузка обучающих ответов."""
    answers = pd.read_csv(PATH / "train_answers.csv").set_index("itemId")
    answers.columns = COORDINATES
    return answers


def test_x():
    """Загрузка тестовых прогнозов."""
    return pd.read_csv(PATH / "test_data.csv").set_index("itemId")
