import unittest
from unittest.mock import MagicMock
import tempfile

from train.train import run
from predict.predict.run import TextPredictionModel
import pandas as pd
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags,
    })


class TestPredict(unittest.TestCase):
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        params = {
                "batch_size": 2,
                "epochs": 5,
                "dense_dim": 64,
                "min_samples_per_label": 1,
                "verbose": 1
            }

        with tempfile.TemporaryDirectory() as model_dir:
            accuracy, path = run.train("fake_path", params, model_dir, "False")

            predictor = TextPredictionModel.from_artefacts(path)

            text_list = [
                "Is it possible to execute the procedure of a function in the scope of the caller?",
                "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            ]

            predictions = predictor.predict(text_list)
            print(predictions)
            self.assertEqual(len(predictions), len(text_list))


