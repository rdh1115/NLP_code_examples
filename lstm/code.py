import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### PROVIDED CODE #####

def tokenize(
        text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]


def build_index_map(
        word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words - 1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}


def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in tokenize(batch['premise'] + batch['hypothesis']):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def tokens_to_ix(
        tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        self.unique_chars = list()
        self.vocab_size = 0
        self.mappings = dict()
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch

        # your code here

        with open(filepath, 'r') as f:
            self.file = f.read()
        self.unique_chars = list(set(self.file))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)

    def generate_char_mappings(self, uq):
        mappings = dict()
        mappings['char_to_idx'] = dict()
        mappings['idx_to_char'] = dict()
        for i, c in enumerate(uq):
            mappings['char_to_idx'][c] = i
            mappings['idx_to_char'][i] = c
        self.mappings = mappings
        return mappings

    def convert_seq_to_indices(self, seq):
        return [self.mappings['char_to_idx'][c] for c in seq]

    def convert_indices_to_seq(self, seq):
        return [self.mappings['idx_to_char'][c] for c in seq]

    def get_example(self):
        def get_random_str(main_str, substr_len):
            idx = random.randrange(0,
                                   len(main_str) - substr_len)
            return main_str[idx: (idx + substr_len)], main_str[idx + 1: (idx + substr_len) + 1]

        for _ in range(self.examples_per_epoch):
            in_seq, out_seq = get_random_str(self.file, self.seq_len)
            in_seq = torch.tensor(self.convert_seq_to_indices(in_seq)).to(device)
            out_seq = torch.tensor(self.convert_seq_to_indices(out_seq)).to(device)
            yield in_seq, out_seq


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars

        self.embedding_size = embedding_size

        # your code here
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.wax = nn.Linear(embedding_size, hidden_size, bias=False)
        self.waa = nn.Linear(hidden_size, hidden_size)
        self.wya = nn.Linear(hidden_size, self.n_chars)

    def rnn_cell(self, i, h):
        h_new = torch.tanh(self.waa(h) + self.wax(i))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(device)

        outs = list()
        for c in input_seq:
            i = self.embedding_layer(c)
            o, hidden = self.rnn_cell(i, hidden)
            outs.append(o)
        out = torch.stack(outs)
        return out, hidden

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        output = [starting_char]

        hidden = None
        for _ in range(seq_len):
            out, hidden = self.forward(torch.tensor([output[-1]]).to(device), hidden)
            out = F.softmax(out.squeeze() / temp, dim=0)
            output.append(Categorical(out).sample().item())
        return output


class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        #  your code here
        concat_size = hidden_size + embedding_size
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.forget_gate = nn.Linear(concat_size, hidden_size)
        self.input_gate = nn.Linear(concat_size, hidden_size)
        self.output_gate = nn.Linear(concat_size, hidden_size)
        self.cell_state_layer = nn.Linear(concat_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, self.n_chars)

    def forward(self, input_seq, hidden=None, cell=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(device)

        if cell is None:
            cell = torch.zeros(self.hidden_size).to(device)

        outs = list()
        for c in input_seq:
            i = self.embedding_layer(c)
            o, hidden, cell = self.lstm_cell(i, hidden, cell)
            outs.append(o)
        out = torch.stack(outs)
        return out, hidden, cell

    def lstm_cell(self, i, h, c):
        concat = torch.concat([i, h])
        forget = torch.sigmoid(self.forget_gate(concat))
        input = torch.sigmoid(self.input_gate(concat))
        output = torch.sigmoid(self.output_gate(concat))

        cell_tilde = torch.tanh(self.cell_state_layer(concat))
        cell_new = forget * c + input * cell_tilde

        hidden_new = output * torch.tanh(cell_new)
        pred = self.fc_output(hidden_new)
        return pred, hidden_new, cell_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        output = [starting_char]

        hidden, cell = None, None
        for _ in range(seq_len):
            out, hidden, cell = self.forward(torch.tensor([output[-1]]).to(device), hidden, cell)
            out = F.softmax(out.squeeze() / temp, dim=0)
            output.append(Categorical(out).sample().item())
        return output


def plot_loss(rnn_losses, lstm_losses):
    import matplotlib.pyplot as plt
    plt.plot(rnn_losses[0], label='RNN Sherlock')
    plt.plot(rnn_losses[0], label='RNN Shakespear')
    plt.plot(lstm_losses[0], label='RNN Sherlock')
    plt.plot(lstm_losses[0], label='RNN Shakespear')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def train(model, dataset, lr, out_seq_len, num_epochs):
    # code to initialize optimizer, loss function

    n = 0
    running_loss = 0
    losses = list()
    model.to(device)

    optimizer = model.get_optimizer(lr)
    loss_f = model.get_loss_function()
    for epoch in range(num_epochs):
        model.train()

        for in_seq, out_seq in dataset.get_example():
            # main loop code
            hidden = None
            output, hidden = model.forward(in_seq, hidden)

            optimizer.zero_grad()
            loss = loss_f(output, out_seq)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += 1
        losses.append(running_loss)
        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss / n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly
        capital_chars = [c for c in dataset.unique_chars if c.isupper()]
        starting_char = random.choice(capital_chars)
        model.eval()
        with torch.no_grad():
            seq = model.sample_sequence(starting_char, out_seq_len)
        print('Generated sequence: ', seq)
        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    return None  # return model optionally


def train_lstm(model, dataset, lr, out_seq_len, num_epochs):
    # code to initialize optimizer, loss function

    n = 0
    running_loss = 0
    model.to(device)

    optimizer = model.get_optimizer(lr)
    loss_f = model.get_loss_function()
    for epoch in range(num_epochs):
        model.train()

        for in_seq, out_seq in dataset.get_example():
            # main loop code
            hidden, cell = None, None
            output, hidden, cell = model.forward(in_seq, hidden, cell)

            optimizer.zero_grad()
            loss = loss_f(output, out_seq)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss / n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly
        capital_chars = [c for c in dataset.unique_chars if c.isupper()]
        starting_char = random.choice(capital_chars)
        model.eval()
        with torch.no_grad():
            seq = model.sample_sequence(starting_char, out_seq_len)
        print('Generated sequence: ', seq)
        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    return None  # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10  # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, epoch_size)

    model = CharRNN(dataset.vocab_size, embedding_size, hidden_size)
    train(model, dataset, lr=lr,
          out_seq_len=out_seq_len,
          num_epochs=num_epochs)


def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, epoch_size)
    model = CharLSTM(dataset.vocab_size, embedding_size, hidden_size)
    train_lstm(model, dataset, lr=lr,
               out_seq_len=out_seq_len,
               num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):
    batch_premises = [torch.tensor(batch).to(device) for batch in batch_premises]
    batch_hypotheses = [torch.tensor(batch).to(device) for batch in batch_hypotheses]

    premises = nn.utils.rnn.pad_sequence(batch_premises, True, 0)
    hypotheses = nn.utils.rnn.pad_sequence(batch_hypotheses, True, 0)

    reverse_batch_premises = [torch.flip(t, dims=[0]) for t in batch_premises]
    reverse_premises = nn.utils.rnn.pad_sequence(reverse_batch_premises, True, 0)

    reverse_batch_hypotheses = [torch.flip(t, dims=[0]) for t in batch_hypotheses]
    reverse_hypotheses = nn.utils.rnn.pad_sequence(reverse_batch_hypotheses, True, 0)

    return premises, hypotheses, reverse_premises, reverse_hypotheses


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    embeddings = torch.empty((len(word_index), emb_dim))
    for s, i in word_index.items():
        if s in emb_dict:
            embeddings[i] = torch.from_numpy(emb_dict[s])
        elif s == "[PAD]":
            embeddings[i] = torch.zeros((emb_dim)).to(device)
        else:
            embeddings[i] = torch.randn((emb_dim)).to(device)
    return embeddings


def evaluate(model, dataloader, index_map):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            premise, hypo = batch['premise'], batch['hypothesis']
            label = batch['label'].to(device)
            premise, hypo = tokens_to_ix(tokenize(premise), index_map), tokens_to_ix(tokenize(hypo), index_map)
            y = model.forward(premise, hypo)
            preds = torch.softmax(y, 1).argmax(1)

            total += len(label)
            correct += (preds == label).sum().item()
    return correct / total


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes, embed_dim):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # your code here
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.int_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        premises, hypotheses, _, _ = fix_padding(a, b)
        p_embed, h_embed = self.embedding_layer(premises), self.embedding_layer(hypotheses)
        _, (_, p_cell) = self.lstm(p_embed)
        _, (_, h_cell) = self.lstm(h_embed)
        concat = torch.concat((p_cell[-1, :, :], h_cell[-1, :, :]), 1)
        X = self.int_layer(concat)
        X = nn.functional.relu(X)
        y = self.out_layer(X)
        return y

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes, embed_dim):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # your code here
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.lstm_forward = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                    batch_first=True)
        self.lstm_backward = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                     batch_first=True)
        self.int_layer = nn.Linear(4 * hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        premises, hypotheses, p_back, h_back = fix_padding(a, b)
        p_embed, h_embed = self.embedding_layer(premises), self.embedding_layer(hypotheses)
        p_back_embed, h_back_embed = self.embedding_layer(p_back), self.embedding_layer(h_back)

        _, (_, p_cell) = self.lstm_forward(p_embed)
        _, (_, h_cell) = self.lstm_forward(h_embed)
        _, (_, p_cell_back) = self.lstm_backward(p_back_embed)
        _, (_, h_cell_back) = self.lstm_backward(h_back_embed)

        concat = torch.concat((p_cell[-1, :, :], p_cell_back[-1, :, :], h_cell[-1, :, :], h_cell_back[-1, :, :]), 1)
        X = self.int_layer(concat)
        X = nn.functional.relu(X)
        y = self.out_layer(X)
        return y

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)


class TrueBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes, embed_dim):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # your code here
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.int_layer = nn.Linear(4 * hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        premises, hypotheses, _, _ = fix_padding(a, b)
        p_embed, h_embed = self.embedding_layer(premises), self.embedding_layer(hypotheses)
        _, (_, p_cell) = self.lstm(p_embed)
        _, (_, h_cell) = self.lstm(h_embed)
        concat = torch.concat((p_cell[-1, :, :], p_cell[-2, :, :], h_cell[-1, :, :], h_cell[-2, :, :]), 1)
        X = self.int_layer(concat)
        X = nn.functional.relu(X)
        y = self.out_layer(X)
        return y

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)


def train_snli(model, dataloader, lr, num_epochs, index_map):
    # code to initialize optimizer, loss function

    n = 0
    running_loss = 0

    accs = list()

    optimizer = model.get_optimizer(lr)
    loss_f = model.get_loss_function()
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()

        for i, batch in enumerate(dataloader):
            # main loop code
            premise, hypo = batch['premise'], batch['hypothesis']
            premise, hypo = tokens_to_ix(tokenize(premise), index_map), tokens_to_ix(tokenize(hypo), index_map)

            y = model.forward(premise, hypo)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            loss = loss_f(y, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += 1
            # print info every X examples
            if i % 200 == 199:
                print(f"Epoch {epoch}, mini batch {i}. Running loss so far: {(running_loss / n):.8f}")
                break
        n = 0
        running_loss = 0

        print("\n-------------MODEL EVAL-------------")

        acc = evaluate(model, dataloader, index_map)
        print(f'--- Accuracy: {acc} ---')
        accs.append(acc)

    print("\n--- %s seconds ---" % (time.time() - start_time))
    return model, accs


def run_snli(model, dataset, glove):
    lr = 0.002
    model.to(device)
    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered = dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders
    dataloader_train = DataLoader(train_filtered, 32, True, num_workers=2)
    dataloader_valid = DataLoader(valid_filtered, 32, True, num_workers=2)
    dataloader_test = DataLoader(test_filtered, 32, True, num_workers=2)
    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    if glove:
        glove = pd.read_csv('/Users/markbai/Documents/School/COMP599/a3_code/data/glove.6B.100d.txt', sep=" ",
                            quoting=3,
                            header=None, index_col=0)
        embed_dict = {s: embed for s, embed in zip(glove.index.values, glove.to_numpy())}
        glove_embeddings = create_embedding_matrix(index_map, embed_dict, 100)
        embeddings = nn.Embedding.from_pretrained(glove_embeddings, freeze=False, padding_idx=0)
        model.embedding_layer = embeddings.to(device)

    # training code
    print('\n--- Training ---')
    model, accs_train = train_snli(model, dataloader_train, lr, 5, index_map)
    print('\n--- Validation ---')
    _, accs_valid = train_snli(model, dataloader_valid, lr, 5, index_map)
    print('\n--- Testing ---')
    _, accs_test = train_snli(model, dataloader_test, lr, 5, index_map)

    return accs_valid, accs_test


def run_snli_lstm(glove, dataset):
    model_class = UniLSTM(35495, 100, 1, 3, 100)
    return run_snli(model_class, dataset, glove)


def run_snli_bilstm(glove, dataset):
    model_class = ShallowBiLSTM(35495, 50, 1, 3, 100)
    return run_snli(model_class, dataset, glove)


def run_snli_true_bilstm(glove, dataset):
    model_class = TrueBiLSTM(35495, 50, 2, 3, 100)
    return run_snli(model_class, dataset, glove)


if __name__ == '__main__':
    pass
    # run_char_rnn()
    # run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
    # dataset = load_dataset("snli")
    # accs = list()
    # accs.append(run_snli_lstm(True, dataset))
    # accs.append(run_snli_lstm(False, dataset))
    # accs.append(run_snli_bilstm(True, dataset))
    # accs.append(run_snli_bilstm(False, dataset))
    # accs.append(run_snli_true_bilstm(True, dataset))
