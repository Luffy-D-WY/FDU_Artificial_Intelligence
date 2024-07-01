import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def read_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, tag = line.split()
                sentence.append((word, tag))
        if sentence:
            sentences.append(sentence)
    return sentences
class NERDataset(Dataset):
    def __init__(self, sentences, word2idx, tag2idx):
        self.sentences = sentences
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [self.word2idx[word] if word in self.word2idx else self.word2idx["<UNK>"] for word, tag in sentence]
        tags = [self.tag2idx[tag] for word, tag in sentence]
        return torch.tensor(words, dtype=torch.long), torch.tensor(tags, dtype=torch.long)
def pad_collate_fn(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=-1)
    return xx_pad, yy_pad, x_lens
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix["<START>"]] = 0.

        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix["<START>"]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
def train_and_predict(language):
    if language == "English":
        train_file = "English/train.txt"
        validation_file = "English/test.txt"
        output_file = "example_data/english_my_result.txt"
    else:
        train_file = "Chinese/train.txt"
        validation_file = "Chinese/validation.txt"
        output_file = "example_data/chinese_my_result.txt"

    # 读取训练和验证数据
    train_sents = read_data(train_file)
    validation_sents = read_data(validation_file)

    # 创建词汇表和标签字典
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    tag2idx = {"<START>": 0, "<STOP>": 1}
    for sentence in train_sents + validation_sents:
        for word, tag in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)

    # 超参数设置
    embedding_dim = 100
    hidden_dim = 128
    learning_rate = 0.005
    batch_size = 1
    num_epochs =1

    # 创建数据集和数据加载器
    train_dataset = NERDataset(train_sents, word2idx, tag2idx)
    validation_dataset = NERDataset(validation_sents, word2idx, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=pad_collate_fn)

    # 定义模型和优化器
    model = BiLSTM_CRF(len(word2idx), tag2idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        model.train()
        for sentences, tags, lengths in train_loader:
            model.zero_grad()
            sentences = sentences.squeeze(0)
            tags = tags.squeeze(0)
            loss = model.neg_log_likelihood(sentences, tags)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



    # model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")


    predictions = []
    with torch.no_grad():
        for sentences, tags, lengths in validation_loader:
            sentences = sentences.squeeze(0)
            score, predicted_tags = model(sentences)
            predictions.append(predicted_tags)

    # 写入预测结果
    def predict_and_write_result(validation_sents, predictions, output_file, tag2idx):
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence, predicted_tags in zip(validation_sents, predictions):
                for (word, _), tag_idx in zip(sentence, predicted_tags):
                    tag = list(tag2idx.keys())[list(tag2idx.values()).index(tag_idx)]
                    f.write(f"{word} {tag}\n")
                f.write("\n")

    predict_and_write_result(validation_sents, predictions, output_file, tag2idx)
    print(f"Prediction completed and results written to {output_file}")

if __name__ == "__main__":
    language = "English"
    model_path = "bilstm_crf_model.pth"
    train_and_predict(language)