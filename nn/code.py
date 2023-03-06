from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
        text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
        word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words - 1]

    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
        tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(
        data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    # TODO: Your code here
    dict_keys = data_dict.keys()
    dict_values = data_dict.values()
    n = len(list(dict_values)[0])

    def loader():
        # TODO: Your code here
        if not shuffle:
            for i in range(0, n, batch_size):
                batch_values = [v[i:i + batch_size] for v in dict_values]
                yield {k: v for k, v in zip(dict_keys, batch_values)}
        else:
            idces = torch.randperm(n)
            batch = [0] * batch_size
            idx_batch = 0
            for idx in idces:
                batch[idx_batch] = idx
                idx_batch += 1
                if idx_batch == batch_size:
                    batch_values = [[v[i] for i in batch] for v in dict_values]
                    yield {k: v for k, v in zip(dict_keys, batch_values)}
                    idx_batch = 0
                    batch = [0] * batch_size
            if idx_batch > 0:
                batch_values = [[v[i] for i in batch[:idx_batch]] for v in dict_values]
                yield {k: v for k, v in zip(dict_keys, batch_values)}

    return loader


### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    # TODO: Your code here
    t = [torch.tensor(l, dtype=torch.int32) for l in text_indices]
    t = nn.utils.rnn.pad_sequence(t, batch_first=True)
    return t


### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, 1).values


class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()

        # TODO: Your code here
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.layer_pred = nn.Linear(embedding.embedding_dim * 2, 1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        # TODO: Your code here
        pre_emd = max_pool(emb(premise))
        hypo_emd = max_pool(emb(hypothesis))
        cat = torch.cat((pre_emd, hypo_emd), 1)
        output = sigmoid(layer_pred(cat))
        output = output.squeeze(1)
        return output


### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    # TODO: Your code here
    return torch.optim.Adam(model.parameters(), **kwargs)


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # TODO: Your code here
    return -(y_pred.log() * y + (1 - y) * (1 - y_pred).log()).mean()


### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    # TODO: Your code here
    torch_device = torch.device(device)
    model.to(torch_device)

    train_batch = dict()
    train_batch['premise'] = convert_to_tensors(batch['premise']).to(torch_device)
    train_batch['hypothesis'] = convert_to_tensors(batch['hypothesis']).to(torch_device)
    y = model(premise=train_batch['premise'], hypothesis=train_batch['hypothesis'])
    return y


def backward_pass(
        optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    # TODO: Your code here
    optimizer.zero_grad()

    loss = bce_loss(y, y_pred)
    loss.backward()
    optimizer.step()
    return loss


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    # TODO: Your code here
    if threshold:
        y_pred = torch.tensor([1 if pred > threshold else 0 for pred in y_pred])
    tp = (y * y_pred).sum().to(torch.float32)
    tn = ((1 - y) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y) * y_pred).sum().to(torch.float32)
    fn = (y * (1 - y_pred)).sum().to(torch.float32)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


### 2.5 Train loop
def eval_run(
        model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):
    # TODO: Your code here
    t_device = torch.device(device)
    y_preds, y_s = torch.empty(0, dtype=torch.float, device=t_device), torch.empty(0, dtype=torch.float,
                                                                                   device=t_device)

    model.eval()
    with torch.no_grad():
        for batch in loader():
            y_preds = torch.cat((y_preds, forward_pass(model, batch, device)), 0)
            y_s = torch.cat((y_s, torch.tensor(batch['label'], dtype=torch.float, device=t_device)), 0)
    return y_s, y_preds


def train_loop(
        model: nn.Module,
        train_loader,
        valid_loader,
        optimizer,
        n_epochs: int = 3,
        device: str = "cpu",
):
    # TODO: Your code here
    t_device = torch.device(device)
    f1_scores = list()

    for epoch in range(n_epochs):
        model.train()

        y_preds, y_s = torch.empty(0, dtype=torch.float, device=t_device), torch.empty(0, dtype=torch.float,
                                                                                       device=t_device)
        for batch in train_loader():
            y_preds = torch.cat((y_preds, forward_pass(model, batch, device)), 0)
            y_s = torch.cat((y_s, torch.tensor(batch['label'], dtype=torch.float, device=t_device)), 0)
        backward_pass(optimizer, y_s, y_preds)

        eval_y_s, eval_y_preds = eval_run(model, valid_loader, device)
        with torch.no_grad():
            f1_scores.append(f1_score(eval_y_s, eval_y_preds))
            torch.save(model.state_dict(), f'model_params_{epoch}')
    return f1_scores


### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()

        # TODO: continue here
        self.embedding = embedding

        self.ff_layer = nn.Linear(embedding.embedding_dim * 2, hidden_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer_pred = nn.Linear(hidden_size, 1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        # TODO: continue here
        pre_emd = max_pool(emb(premise))
        hypo_emd = max_pool(emb(hypothesis))
        cat = torch.cat((pre_emd, hypo_emd), 1)
        linear_act = act(ff_layer(cat))
        output = sigmoid(layer_pred(linear_act))
        output = output.squeeze(1)
        return output


### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()

        # TODO: continue here
        self.embedding = embedding

        self.layer_pred = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        layers = [nn.Linear(embedding.embedding_dim * 2, hidden_size)] + [
            nn.Linear(hidden_size, hidden_size)] * (num_layers - 1)
        self.ff_layers = nn.ModuleList(layers)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        # TODO: continue here
        pre_emd = max_pool(emb(premise))
        hypo_emd = max_pool(emb(hypothesis))
        x = torch.cat((pre_emd, hypo_emd), 1)
        for l in ff_layers:
            x = act(l(x))
        output = sigmoid(layer_pred(x))
        output = output.squeeze(1)
        return output


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }

    # 1.1
    train_loader = build_loader(train_indices)
    valid_loader = build_loader(valid_indices)

    # 1.2
    batch = next(train_loader())
    y = "your code here"

    # 2.1
    embedding = "your code here"
    model = "your code here"

    # 2.2
    optimizer = "your code here"

    # 2.3
    y_pred = "your code here"
    loss = "your code here"

    # 2.4
    score = "your code here"

    # 2.5
    n_epochs = 2

    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.1
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.2
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"
