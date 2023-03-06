import random
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers


# ######################## PART 1: PROVIDED CODE ########################

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


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        dd = data_dict

        if len(dd["premise"]) != len(dd["hypothesis"]) or len(dd["premise"]) != len(
                dd["label"]
        ):
            raise AttributeError("Incorrect length in data_dict")

    def __len__(self):
        return len(self.data_dict["premise"])

    def __getitem__(self, idx):
        dd = self.data_dict
        return dd["premise"][idx], dd["hypothesis"][idx], dd["label"][idx]


def train_distilbert(model, loader, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
        target = target.to(device, dtype=torch.float32)

        pred = model(inputs)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_distilbert(model, loader, device):
    model.eval()

    targets = []
    preds = []

    for premise, hypothesis, target in loader:
        preds.append(model(model.tokenize(premise, hypothesis).to(device)))

        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


# ######################## PART 1: YOUR WORK STARTS HERE ########################
class CustomDistilBert(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: your work below
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.pred_layer = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    # vvvvv DO NOT CHANGE BELOW THIS LINE vvvvv
    def get_distilbert(self):
        return self.distilbert

    def get_tokenizer(self):
        return self.tokenizer

    def get_pred_layer(self):
        return self.pred_layer

    def get_sigmoid(self):
        return self.sigmoid

    def get_criterion(self):
        return self.criterion

    # ^^^^^ DO NOT CHANGE ABOVE THIS LINE ^^^^^

    def assign_optimizer(self, **kwargs):
        return torch.optim.Adam(self.parameters(), **kwargs);

    def slice_cls_hidden_state(
            self, x: transformers.modeling_outputs.BaseModelOutput
    ) -> torch.Tensor:
        return x.last_hidden_state[:, 0]

    def tokenize(
            self,
            premise: "list[str]",
            hypothesis: "list[str]",
            max_length: int = 128,
            truncation: bool = True,
            padding: bool = True,
    ):
        tokenizer = self.get_tokenizer()
        return tokenizer(premise, hypothesis, max_length=max_length, truncation=truncation,
                         padding=padding, return_tensors='pt')

    def forward(self, inputs: transformers.BatchEncoding):
        output = self.get_distilbert()(**inputs)
        cls_hs = self.slice_cls_hidden_state(output)
        preds = self.get_sigmoid()(self.pred_layer(cls_hs))
        return preds.squeeze(1)


# ######################## PART 2: YOUR WORK HERE ########################
def freeze_params(model):
    for param in model.base_model.parameters():
        param.requires_grad = False
    return


def pad_attention_mask(mask, p):
    return F.pad(mask, (p, 0), value=1)


class SoftPrompting(nn.Module):
    def __init__(self, p: int, e: int):
        super().__init__()
        self.p = p
        self.e = e

        self.prompts = torch.randn((p, e), requires_grad=True)

    def forward(self, embedded):
        prompts = self.prompts.repeat(embedded.size(0), 1, 1)
        return torch.concat((prompts, embedded), 1)


# ######################## PART 3: YOUR WORK HERE ########################

def load_models_and_tokenizer(q_name, a_name, t_name, device='cpu'):
    q_enc = transformers.AutoModel.from_pretrained(q_name).to(device)
    a_enc = transformers.AutoModel.from_pretrained(a_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(t_name)
    return q_enc, a_enc, tokenizer


def tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers, max_length=64) -> transformers.BatchEncoding:
    q_batch = tokenizer(q_titles, q_bodies, max_length=max_length, truncation=True,
                        padding=True, return_tensors='pt')
    a_batch = tokenizer(answers, max_length=max_length, truncation=True,
                        padding=True, return_tensors='pt')
    return q_batch, a_batch


def get_class_output(model, batch):
    return model(**batch).last_hidden_state[:, 0]


def inbatch_negative_sampling(Q: Tensor, P: Tensor, device: str = 'cpu') -> Tensor:
    S = torch.matmul(Q, torch.transpose(P, 0, 1)).to(device)
    return S


def contrastive_loss_criterion(S: Tensor, labels: Tensor = None, device: str = 'cpu'):
    if labels is None:
        labels = torch.arange(end=S.size(0))
    labels = labels.to(device)
    softmax_S = F.log_softmax(S, dim=1)

    loss = F.nll_loss(softmax_S, labels)
    return loss


def get_topk_indices(Q, P, k: int = None):
    scores = inbatch_negative_sampling(Q, P)
    sorted_scores, sorted_idx = torch.sort(scores, dim=1, descending=True)
    indices, scores = sorted_idx[:, 0:k], sorted_scores[:, 0:k]
    return indices, scores


def select_by_indices(indices: Tensor, passages: 'list[str]') -> 'list[str]':
    return [[passages[a] for a in idx_list] for idx_list in indices]


def embed_passages(passages: 'list[str]', model, tokenizer, device='cpu', max_length=512):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(passages, max_length=max_length, truncation=True,
                           padding=True, return_tensors='pt').to(device)
        return get_class_output(model, inputs)


def embed_questions(titles, bodies, model, tokenizer, device='cpu', max_length=512):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(titles, bodies, max_length=max_length, truncation=True,
                           padding=True, return_tensors='pt').to(device)
        return get_class_output(model, inputs)


def recall_at_k(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]', k: int):
    correct = 0
    for idx, retrieved_idx in zip(true_indices, retrieved_indices):
        if idx in retrieved_idx[0:k]:
            correct += 1
    return correct / len(true_indices)


def mean_reciprocal_rank(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]'):
    rank = 0
    for idx, retrieved_idx in zip(true_indices, retrieved_indices):
        if idx in retrieved_idx:
            rank += 1 / (retrieved_idx.index(idx) + 1)
    return rank / len(true_indices)


# ######################## PART 4: YOUR WORK HERE ########################


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import accuracy_score  # Make sure sklearn is installed

    random.seed(2022)
    torch.manual_seed(2022)

    # Parameters (you can change them)
    sample_size = None  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 10
    num_words = 50000

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data/nli")
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
            train_raw["premise"]
            + train_raw["hypothesis"]
            + valid_raw["premise"]
            + valid_raw["hypothesis"]
    )

    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    train_loader = torch.utils.data.DataLoader(
        NLIDataset(train_raw), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        NLIDataset(valid_raw), batch_size=batch_size, shuffle=False
    )

    model = CustomDistilBert().to(device)
    optimizer = model.assign_optimizer(lr=1e-4)

    acc_train = list()
    acc_valid = list()

    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, device=device)

        preds_train, targets_train = eval_distilbert(model, train_loader, device=device)
        preds_train = preds_train.round()
        train_score = accuracy_score(targets_train.cpu(), preds_train.cpu())
        acc_train.append(train_score)

        preds_valid, targets_valid = eval_distilbert(model, valid_loader, device=device)
        preds_valid = preds_valid.round()
        valid_score = accuracy_score(targets_valid.cpu(), preds_valid.cpu())
        acc_valid.append(valid_score)

        print("Epoch:", epoch)
        print("Training loss:", loss)
        print("Validation F1 score:", valid_score)
        print()
