import random
import unittest
import code
import torch


class TestLoader(unittest.TestCase):
    def test_build_loader(self):
        data = {
            'h': [1, 2, 3, 4],
            'p': ['a', 'b', 'c', 'd'],
        }
        gen = code.build_loader(data, 2, True)
        for batch in gen():
            print(batch)
        gen = code.build_loader(data, 2, False)
        for batch in gen():
            print(batch)

    def test_tensor(self):
        data = list()
        data.append([1, 2, 3, 4, 5])
        data.append([6, 7, 8, 9, 10])
        data.append([11, 12, 13, 14])
        print(code.convert_to_tensors(data))


class TestLogistic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLogistic, self).__init__(*args, **kwargs)
        # Seeds to ensure reproducibility
        random.seed(2022)
        torch.manual_seed(2022)

        # If you use GPUs, use the code below:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prefilled code showing you how to use the helper functions
        train_raw, valid_raw = code.load_datasets("data")

        train_tokens = {
            "premise": code.tokenize(train_raw["premise"], max_length=64),
            "hypothesis": code.tokenize(train_raw["hypothesis"], max_length=64),
        }

        valid_tokens = {
            "premise": code.tokenize(valid_raw["premise"], max_length=64),
            "hypothesis": code.tokenize(valid_raw["hypothesis"], max_length=64),
        }

        word_counts = code.build_word_counts(
            train_tokens["premise"]
            + train_tokens["hypothesis"]
            + valid_tokens["premise"]
            + valid_tokens["hypothesis"]
        )
        index_map = code.build_index_map(word_counts, max_words=10000)

        train_indices = {
            "label": train_raw["label"],
            "premise": code.tokens_to_ix(train_tokens["premise"], index_map),
            "hypothesis": code.tokens_to_ix(train_tokens["hypothesis"], index_map)
        }

        valid_indices = {
            "label": valid_raw["label"],
            "premise": code.tokens_to_ix(valid_tokens["premise"], index_map),
            "hypothesis": code.tokens_to_ix(valid_tokens["hypothesis"], index_map)
        }

        # 1.1
        self.train_loader = code.build_loader(train_indices)
        self.valid_loader = code.build_loader(valid_indices)

    def test_max(self):
        data = list()
        data.append([[1, 2, 3], [2, 3, 4]])
        data.append([[1, 2, 3], [4, 5, 6]])
        data.append([[1, 2, 3], [5, 6, 7]])
        t = code.convert_to_tensors(data)
        print(code.max_pool(t))

    def test_forward(self):
        data = {'premise': [[7, 44, 15, 45, 10, 25, 4, 46, 26], [7, 44, 15, 45, 10, 25, 4, 46, 26],
                            [7, 20, 47, 2, 11, 27, 21, 8, 3, 22, 12, 21, 8, 3, 22, 15, 16, 17, 48, 49, 2, 1, 50, 12,
                             51, 28, 29, 2, 1, 52]],
                'hypothesis': [[7, 9, 15, 25, 26], [3, 82, 15, 83, 84, 1, 85], [7, 81, 2, 86, 27, 87, 28, 29]],
                'label': [0, 1, 0]
                }
        model = code.PooledLogisticRegression(torch.nn.Embedding(100, 100))
        print(code.forward_pass(model, data))

    def test_eval(self):
        data = [{'premise': [[7, 44, 15, 45, 10, 25, 4, 46, 26], [7, 44, 15, 45, 10, 25, 4, 46, 26],
                             [7, 20, 47, 2, 11, 27, 21, 8, 3, 22, 12, 21, 8, 3, 22, 15, 16, 17, 48, 49, 2, 1, 50, 12,
                              51, 28, 29, 2, 1, 52]],
                 'hypothesis': [[7, 9, 15, 25, 26], [3, 82, 15, 83, 84, 1, 85], [7, 81, 2, 86, 27, 87, 28, 29]],
                 'label': [0, 1, 0]}, {'premise': [
            [7, 20, 47, 2, 11, 27, 21, 8, 3, 22, 12, 21, 8, 3, 22, 15, 16, 17, 48, 49, 2, 1, 50, 12, 51, 28, 29, 2, 1,
             52], [1, 13, 30, 31, 4, 1, 32, 53, 1, 54, 55, 56, 57, 2, 3, 58, 5, 59],
            [1, 13, 30, 31, 4, 1, 32, 53, 1, 54, 55, 56, 57, 2, 3, 58, 5, 59]],
                                       'hypothesis': [[7, 81, 2, 88, 89, 4, 90], [1, 9, 91, 43, 92, 2, 1, 93, 94],
                                                      [1, 13, 30, 31, 4, 1, 32]], 'label': [1, 1, 0]}, {
                    'premise': [[7, 20, 33, 5, 60, 61, 34, 35, 10, 14, 62, 63, 64, 12, 65],
                                [7, 20, 33, 5, 60, 61, 34, 35, 10, 14, 62, 63, 64, 12, 65],
                                [1, 13, 2, 1, 11, 23, 16, 2, 66, 5, 1, 67, 68, 69, 8, 70, 71]],
                    'hypothesis': [[33, 34, 35], [95, 96, 97, 5, 98], [1, 13, 6, 14, 1, 11, 23]], 'label': [0, 1, 0]}, {
                    'premise': [[1, 13, 2, 1, 11, 23, 16, 2, 66, 5, 1, 67, 68, 69, 8, 70, 71],
                                [72, 1, 11, 73, 8, 74, 75, 1, 24, 76, 77, 4, 36, 3, 37, 5, 1, 38, 16, 78, 17, 1, 79, 5,
                                 80],
                                [72, 1, 11, 73, 8, 74, 75, 1, 24, 76, 77, 4, 36, 3, 37, 5, 1, 38, 16, 78, 17, 1, 79, 5,
                                 80]],
                    'hypothesis': [[1, 13, 6, 14, 1, 99, 23], [1, 24, 6, 100, 4, 36, 3, 37, 5, 1, 38],
                                   [1, 24, 6, 101, 8, 1, 102]], 'label': [1, 0, 1]}, {
                    'premise': [[1, 9, 6, 18, 1, 19, 10, 14, 1, 39, 2, 3, 40, 41, 4, 3, 42],
                                [1, 9, 6, 18, 1, 19, 10, 14, 1, 39, 2, 3, 40, 41, 4, 3, 42],
                                [1, 9, 6, 18, 1, 19, 10, 14, 1, 39, 2, 3, 40, 41, 4, 3, 42]],
                    'hypothesis': [[1, 9, 6, 18, 1, 19], [1, 9, 6, 103, 43, 104],
                                   [1, 9, 6, 18, 1, 19, 12, 105, 17, 43, 106]], 'label': [0, 1, 1]}]
        def loader():
            yield from data

        model = code.PooledLogisticRegression(torch.nn.Embedding(500, 100))
        code.eval_run(model, loader)

    def test_training(self):
        model = code.PooledLogisticRegression(torch.nn.Embedding(10000, 100))
        adam = code.assign_optimizer(model)
        print(code.train_loop(model, self.train_loader, self.valid_loader, adam))

if __name__ == '__main__':
    unittest.main()
