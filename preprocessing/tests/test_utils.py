#from curses.ascii import BS
import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methodspip
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_train_samples = MagicMock(return_value=100)
        # we assert that _get_num_test_samples will return 100 / batch_size = 5
        self.assertEqual(base._get_num_train_batches(), 5)

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_test_samples = MagicMock(return_value=100)
        # we assert that _get_num_test_batches will return 100 / batch_size = 5
        self.assertEqual(base._get_num_test_batches(), 5)

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["Python", "Java", "PHP"])

        self.assertEqual(base.get_index_to_label_map(), {0: "Python", 1: "Java", 2: "PHP"})

    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["Python", "Java", "PHP"])

        # we assert that index_to_label keys = label_to_index values and vice versa
        self.assertEqual(list(base.get_index_to_label_map().keys()), list(base.get_label_to_index_map().values()))
        self.assertEqual(list(base.get_index_to_label_map().values()), list(base.get_label_to_index_map().keys()))

    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value={'Python': 0, 'PHP': 1})
        labels = ['Python', 'PHP', 'Python']
        self.assertEqual(base.to_indexes(labels), [0, 1, 0])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_1', 'id_2', 'id_1', 'id_2', 'id_1', 'id_2', 'id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', ],
            'tag_id': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, ],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10', ]
        }))

        base = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1)
        self.assertEqual(base._get_num_samples(), 10)

    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", 2, 0.5, 1)
        x_train, y_train = dataset.get_train_batch()
        self.assertEqual(x_train.shape, (2,))
        self.assertEqual(y_train.shape, (2, 2))

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", 2, 0.5, 1)
        x_test, y_test = dataset.get_test_batch()
        self.assertEqual(x_test.shape, (2,))
        self.assertEqual(y_test.shape, (2, 2))

    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9',
                      'title_10']
        }))
        with self.assertRaises(AssertionError):
            txt_cat = utils.LocalTextCategorizationDataset("fake_path", 7, train_ratio=0.5,
                                                           min_samples_per_label=1)

