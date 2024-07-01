import sklearn_crfsuite
from sklearn_crfsuite import metrics

# 读取数据函数
def read_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                char, label = line.split()
                sentence.append((char, label))
        if sentence:
            sentences.append(sentence)
    return sentences

# 特征提取函数
def char2features(sent, i):
    char = sent[i][0]
    features = {
        'bias': 1.0,
        'char': char,
        'char.isdigit()': char.isdigit(),
        'char.isupper()': char.isupper(),
        'char.islower()': char.islower(),
    }
    if i > 0:
        char1 = sent[i-1][0]
        features.update({
            '-1:char': char1,
            '-1:char.isdigit()': char1.isdigit(),
            '-1:char.isupper()': char1.isupper(),
            '-1:char.islower()': char1.islower(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        char1 = sent[i+1][0]
        features.update({
            '+1:char': char1,
            '+1:char.isdigit()': char1.isdigit(),
            '+1:char.isupper()': char1.isupper(),
            '+1:char.islower()': char1.islower(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [char2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for char, label in sent]

# 训练模型函数
def train_crf(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )
    crf.fit(X_train, y_train)
    return crf

# 预测并保存结果函数
def predict_and_save_results(crf, test_sents, X_test, output_file):
    y_pred = crf.predict(X_test)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent, preds in zip(test_sents, y_pred):
            for (char, _), pred_label in zip(sent, preds):
                f.write(f"{char} {pred_label}\n")
            f.write("\n")

# 主函数
def main(language):
    if language == "English":
        train_file = "English/train.txt"
        validation_file = "English/test.txt"
        output_file = "example_data/2english_my_result.txt"
    else:
        train_file = "Chinese/train.txt"
        validation_file = "Chinese/test.txt"
        output_file = "example_data/2chinese_my_result.txt"

    # 读取训练和测试数据
    train_sents = read_data(train_file)
    test_sents = read_data(validation_file)

    # 提取特征和标签
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # 训练模型
    crf = train_crf(X_train, y_train)

    # 预测并保存结果
    predict_and_save_results(crf, test_sents, X_test, output_file)
    print(f"Prediction completed and results written to {output_file}")

if __name__ == "__main__":
    language = "English"
    main(language)
    language="Chinese"
    main(language)
