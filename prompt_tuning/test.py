import random
import unittest
import code
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn

class TestCustomBertLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCustomBertLoader, self).__init__(*args, **kwargs)

    def testGetExample(self):
        premise, hypothesis = ['hi','hello'], ['bye','world']
        model = code.CustomDistilBert()
        inputs = model.tokenize(premise, hypothesis)
        print(model.forward(inputs))

    def testGetBatch(self):
        train_raw, valid_raw = code.load_datasets("data/nli")
        valid_loader = torch.utils.data.DataLoader(
            code.NLIDataset(valid_raw), batch_size=1, shuffle=True
        )
        print([next(iter(valid_loader)) for _ in range(3)])


if __name__ == '__main__':
    unittest.main()
