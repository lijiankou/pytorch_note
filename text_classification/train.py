import os
import sys
sys.path.insert(0, '../tool')
import log

import os
import logging
import argparse

import torch
import sys

from torchtext.datasets import text_classification
from torch.utils.data import DataLoader

from model import TextSentiment
from torch.utils.data.dataset import random_split

def ArgParse():
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='num epochs (default=5)')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='embed dim. (default=32)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--split-ratio', type=float, default=0.95,
                        help='train/valid split ratio (default=0.95)')
    parser.add_argument('--lr', type=float, default=4.0,
                        help='learning rate (default=4.0)')
    parser.add_argument('--lr-gamma', type=float, default=0.8,
                        help='gamma value for lr (default=0.8)')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='ngrams (default=2)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='num of workers (default=1)')
    parser.add_argument('--device', default='cpu',
                        help='device (default=cpu)')
    parser.add_argument('--data', default='.data',
                        help='data directory (default=.data)')
    parser.add_argument('--use-sp-tokenizer', type=bool, default=False,
                        help='use sentencepiece tokenizer (default=False)')
    parser.add_argument('--sp-vocab-size', type=int, default=20000,
                        help='vocab size in sentencepiece model (default=20000)')
    parser.add_argument('--dictionary',
                        help='path to save vocab')
    parser.add_argument('--save-model-path',
                        help='path for saving model')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    args = parser.parse_args()
    return args


r"""
This file shows the training process of the text classification model.
"""


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.

    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


r"""
torch.utils.data.DataLoader is recommended for PyTorch users to load data.
We use DataLoader here to load datasets and send it to the train_and_valid()
and text() functions.

"""

def TrainValid(num_epochs, num_workers, device, batch_size,
               lr_, lr_gamma, train, valid, model = ''):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_gamma)
    train_data = DataLoader(train, batch_size=batch_size, shuffle=True,
                            collate_fn=generate_batch, num_workers=num_workers)
    num_lines = num_epochs * len(train)
    loss_fun = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(num_epochs):
        for i, (text, offsets, cls) in enumerate(train_data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = loss_fun(output, cls)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, scheduler.get_lr()[0], loss))
        scheduler.step()
        print("Valid - Accuracy: {}".format(test(sub_valid_, model)))

def test(batch_size, data_, model = ''):
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)

    if total_accuracy == []:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)


if __name__ == "__main__":
    logger = log.GetLogger(log.logging.INFO)
    a = ArgParse()

    logging.basicConfig(level=getattr(logging, a.logging_level))

    if not os.path.exists(a.data):
        print("Creating directory {}".format(a.data))
        os.mkdir(data)

    logger.info(a.dataset)
    logger.info(a.data)
    train_data, test_data = text_classification.DATASETS[a.dataset](root=a.data, ngrams=a.ngrams)
    model = TextSentiment(len(train_data.get_vocab()),
                              a.embed_dim, len(train_data.get_labels())).to(a.device)

    train_len = int(len(train_data) * a.split_ratio)
    train, valid = random_split(train_data, [train_len, len(train_data) - train_len])


    TrainValid(a.num_epochs, a.num_worders, a.device, a.batch_size, a.lr, a.lr_gamma,
               train, valid, model = model)
    acc = test(test_data, model)
    print("Test - Accuracy: {}".format(acc))

    if a.save_model_path:
        logger.info(a.save_model_path)
        torch.save(model.to('cpu'), a.save_model_path)

    if a.dictionary is not None:
        print("Save vocab to {}".format(a.dictionary))
        torch.save(train_dataset.get_vocab(), a.dictionary)
