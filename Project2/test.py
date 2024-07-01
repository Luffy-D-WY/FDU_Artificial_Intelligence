import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchtext
from collections import Counter

import model.ChainCRF
import dataLoad.lload
from dataLoad.lload import CustomDataset
from model.ChainCRF import BLITM, ChainCRF
from get_data import get_train_data, get_valid_data
from check import check

# Reload modules
importlib.reload(model.ChainCRF)
importlib.reload(dataLoad.lload)

# Configuration
language = "Chinese"
hidden_dim = 100 if language == "English" else 100
embed_size = 100 if language == "English" else 200
for_test = True
min_freq = 4
mode = False

# BiLSTM_CRF Model Definition
class BiLSTM_CRF(nn.Module):
    def __init__(self, num_classes, vocab_length, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BLITM(num_classes, vocab_length, embedding_dim, hidden_dim)
        self.crf = ChainCRF(num_classes)

    def forward(self, sentence, mask, targets=None, pre_train=None):
        emissions = self.bilstm(sentence)
        if targets is not None:
            crf_loss = self.crf(emissions, targets, mask)
            return crf_loss
        else:
            tags = self.crf.viterbi_decode(emissions, mask)
            return tags

# Data Preparation
train_data = get_train_data(language)
valid_data = get_valid_data(language)

train_word = [word for sentence in train_data for word, label in sentence]
vocab1 = torchtext.vocab.vocab(Counter(train_word), min_freq=min_freq, specials=['<unk>'])
vocab1.set_default_index(vocab1['<unk>'])

train_data += valid_data
train_word = [word for sentence in train_data for word, label in sentence]
vocab2 = torchtext.vocab.vocab(Counter(train_word), min_freq=min_freq, specials=['<unk>'])
vocab2.set_default_index(vocab2['<unk>'])

vocab = vocab2 if for_test else vocab1

def sent2word(sentence):
    return [w for w, _ in sentence]

def sent2label(sentence):
    return [l for _, l in sentence]

max_length = max(max([len(l) for l in train_data]), 256)
sorted_labels = sorted_labels_chn if language == 'Chinese' else sorted_labels_eng

def label2index(label):
    return sorted_labels.index(label)

custom_dataset = CustomDataset(train_data, vocab, label2index, max_length)
batch_size = 32
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Training Function
def train(model, train_loader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(total=num_epochs * len(train_loader))

    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0

        for batch in train_loader:
            length = batch['max_length']
            max_length = np.argmax(length)
            aaa = length[max_length]
            inputs = batch['word_embeddings'][:, :aaa].to(device)
            labels = batch['label_indices'][:, :aaa].to(device)
            mask = batch['mask'].to(device)[:, :aaa].to(device)

            optimizer.zero_grad()
            loss = model(inputs, mask, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            sum_loss += loss.item()
            optimizer.step()
            progress_bar.update(1)

        progress_bar.set_postfix_str(f"Epoch:{epoch + 1}, Loss:{sum_loss / len(train_loader)}")
        print(" ")

    progress_bar.close()

# Example usage
bilstm_crf = BiLSTM_CRF(len(sorted_labels), len(vocab), embed_size, hidden_dim)
pretrain_file = f"./bilstm_crf/pretrain/BILSTM_{language}.bin"
save_file = f"./weight/bilstm/BILSTM_CRF_{language}_final__temp.bin" if for_test else f"./weight/bilstm/BILSTM_CRF_{language}__temp.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_file1 = f"./weight/bilstm/BILSTM_CRF_{language}.bin"
load_file2 = f"./weight/bilstm/BILSTM_CRF_{language}_final.bin"

if mode:
    train(bilstm_crf, dataloader, 10, 1e-3, device)
    torch.save(bilstm_crf.state_dict(), save_file)
else:
    load_file = load_file2 if for_test else load_file1
    bilstm_crf.load_state_dict(torch.load(load_file))

def mycheck(language, vocab, res_file, model, train_or_valid, device):
    valid = get_data_from_file(res_file)
    pred_path = f"example_data/BILSTM_CRF_{language}_{train_or_valid}.txt"
    valid_data = CustomDataset(valid, vocab, label2index, 256)
    valdataloader = DataLoader(valid_data, batch_size=64, shuffle=False)
    model.to(device)

    with open(pred_path, "w", encoding='utf-8') as f:
        with torch.no_grad():
            iter = 0
            for batch in valdataloader:
                length = batch['max_length']
                max_length = np.argmax(length)
                aaa = length[max_length]
                word_embeddings = batch['word_embeddings'][:, :aaa]
                masks = batch['mask'][:, :aaa]
                preds = model(word_embeddings, masks)

                for pred in preds:
                    pred_labels = []
                    for i in range(len(valid[iter])):
                        f.write(f"{valid[iter][i][0]} {sorted_labels[pred[i]]}\n")
                        pred_labels.append(sorted_labels[pred[i]])
                    f.write('\n')
                    iter += 1

    model.to("cpu")
    res = check(language, res_file, pred_path)
    return res

def test(language, res_file, device):
    bilstm_crf2 = BiLSTM_CRF(len(sorted_labels), len(vocab2), embed_size, hidden_dim)
    load_file = save_file if mode else load_file2
    bilstm_crf2.load_state_dict(torch.load(load_file))
    res2 = mycheck(language, vocab2, res_file, bilstm_crf2, "test", device)
    return res2

test(language=language, res_file=f"test/{language}/test.txt", device=device)
