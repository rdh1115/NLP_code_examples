import random
import unittest
import code
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn

class TestCharLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCharLoader, self).__init__(*args, **kwargs)
        self.fp = '/Users/markbai/Documents/School/COMP599/a3_code/data/shakespeare.txt'
        self.charLoader = code.CharSeqDataloader(self.fp, 20, 5)

    def testGetExample(self):
        for in_q, out_q in self.charLoader.get_example():
            print(in_q, out_q)
            print(''.join(self.charLoader.convert_indices_to_seq(in_q.tolist())),
                  ''.join(self.charLoader.convert_indices_to_seq(out_q.tolist())))
        print([c for c in self.charLoader.unique_chars if c.isupper()])

    def testForward(self):
        model = code.CharRNN(self.charLoader.vocab_size, 100, 100)

        model.forward(torch.tensor([1, 2, 3, 4]))

    def testSample(self):
        model = code.CharRNN(self.charLoader.vocab_size, 100, 100)
        print(self.charLoader.convert_indices_to_seq(model.sample_sequence(1, 10)))

    def testPadding(self):
        hypo = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])]
        premise = [torch.tensor([4, 5, 6]), torch.tensor([1, 2]), torch.tensor([3])]
        print(code.fix_padding(premise, hypo).type)

    def testEval(self):
        data = [{'premise': ['A senior is waiting at the window of a restaurant that serves sandwiches.'],
                 'hypothesis': ['A man is looking to order a grilled cheese sandwich.'], 'label': torch.tensor([1])},
                {'premise': [
                    'Man in a black suit, white shirt and black bowtie playing an instrument with the rest of his symphony surrounding him.'],
                    'hypothesis': [
                        'A person in a suit'],
                    'label': torch.tensor(
                        [0])}]
        dataloader = (batch for batch in data)
        idx_map = code.build_index_map(code.build_word_counts(dataloader))

        model = code.UniLSTM(len(idx_map), 512, 1, 3)
        dataloader = (batch for batch in data)
        print(code.evaluate(model, dataloader, idx_map))

    def testVocab(self):
        dataset = load_dataset("snli")
        code.run_snli_lstm(False, dataset)

    def testDataset(self):
        dataset = load_dataset("snli")
        train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
        valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
        test_filtered = dataset['test'].filter(lambda ex: ex['label'] != -1)
        dataloader_train = DataLoader(train_filtered, 32, True)
        dataloader_valid = DataLoader(valid_filtered, 32, True)
        dataloader_test = DataLoader(test_filtered, 32, True)
        word_counts = code.build_word_counts(dataloader_train)
        index_map = code.build_index_map(word_counts)

        for i, batch in enumerate(dataloader_train):
            # main loop code
            premise, hypo = batch['premise'], batch['hypothesis']
            premise, hypo = code.tokens_to_ix(code.tokenize(premise), index_map), code.tokens_to_ix(code.tokenize(hypo), index_map)

            for p, h in zip(premise, hypo):
                if any(type(e)!=int for e in p):
                    print(p)
                if any(type(e)!=int for e in h):
                    print(h)

if __name__ == '__main__':
    unittest.main()
