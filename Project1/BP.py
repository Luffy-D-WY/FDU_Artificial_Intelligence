import numpy as np
import os
import PIL
from PIL import Image
import random
import matplotlib.pyplot as plt
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, neurons_per_layer, output_size,dropout_rate):
        self.weights = []
        self.biases = []
        self.m_weights = []  # Adam动量权重
        self.v_weights = []  # Adam速度权重
        self.m_biases = []  # Adam动量偏置
        self.v_biases = []  # Adam速度偏置
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # 时间步长初始化为0
        self.accuracy_history = []
        self.dropout_rate=dropout_rate
        # 初始化权重和偏置
        for i in range(hidden_layers+1):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = neurons_per_layer[i - 1]
            output_dim = neurons_per_layer[i] if i <= hidden_layers - 1 else output_size
            weight = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
            bias = np.zeros(output_dim)

            self.weights.append(weight)
            self.biases.append(bias)

            # 初始化Adam变量
            self.m_weights.append(np.zeros_like(weight))
            self.v_weights.append(np.zeros_like(weight))
            self.m_biases.append(np.zeros_like(bias))
            self.v_biases.append(np.zeros_like(bias))
    def apply_dropout(self, x):
        # 在训练过程中应用 dropout 操作
        mask = np.random.rand(*x.shape) < self.dropout_rate
        x *= mask
        x /= self.dropout_rate
        return x
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    def plot_accuracy(self):
        plt.plot(self.accuracy_history, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Plot')
        plt.legend()
        plt.show()

    def forward(self, x,training=True):
        activations = []
        layer_output = x

        # 前向传播计算每一层的输出
        for i in range(len(self.weights)):
            layer_output = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                layer_output = self.relu(layer_output)  # 使用 ReLU 激活函数
                if training:
                    layer_output = self.apply_dropout(layer_output)  # 应用 dropout 操作
            activations.append(layer_output)

        # 使用softmax处理最后的输出
        exp_vals = np.exp(layer_output - np.max(layer_output, axis=1, keepdims=True))
        activations[-1] = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        return activations



    def backward(self, x,activations, y_true, learning_rate):
        self.t += 1
        y_pred = activations[-1]
        delta = y_pred - y_true
        d_weights = []
        d_biases = []


        d_weights.append(activations[-2].T.dot(delta))
        d_biases.append(np.sum(delta, axis=0, keepdims=True))

        for i in range(len(self.weights)-1 , 0, -1):
            delta = delta.dot(self.weights[i].T) * np.where(activations[i-1] > 0, 1, 0)
            if i == 1:
                d_weight = x.T.dot(delta)
            else:
                d_weight = activations[i - 2].T.dot(delta)
            d_bias = np.sum(delta, axis=0, keepdims=True)
            d_weights.append(d_weight)
            d_biases.append(d_bias)

        d_weights.reverse()
        d_biases.reverse()



        for i in range(len(self.weights)):
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * d_weights[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (d_weights[i] ** 2)
            m_hat = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_weights[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * d_biases[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (d_biases[i] ** 2)
            m_hat_bias = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_bias = self.v_biases[i] / (1 - self.beta2 ** self.t)

            self.biases[i] -= (learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)).reshape(self.biases[i].shape)


    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-9  # 小常数，防止计算对数0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 将预测值限制在[epsilon, 1-epsilon]区间内
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def train(self, X_train, y_train, learning_rate, epochs):
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # 前向传播以得到网络激活
            activations = self.forward(X_train_shuffled)

            # 反向传播更新网络参数，注意传递X_train_shuffled作为输入
            self.backward(X_train_shuffled, activations, y_train_shuffled, learning_rate)
            #
            # if (epoch + 1) % 10 == 0:
            #      # 计算损失并打印，确保传递正确地预测和真实标签
            #     loss = self.cross_entropy_loss(y_train_shuffled, activations[-1])
            #     print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
            if(epoch+1)%100==0:
                accuracy = evaluate_model(self, X_test, y_test)
                self.accuracy_history.append(accuracy)
                # 打印准确率并在必要时绘制准确率图像
                print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}')
                self.plot_accuracy()
def load_model_parameters(filepath):
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
        biases = pickle.load(f)
    return weights, biases
def evaluate_model(network, X_test, Y_test):
    predictions = network.forward(X_test,False)
    predicted_classes = np.argmax(predictions[-1], axis=1)
    true_classes = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy
    # 保存训练好的参数到本地文件
def save_model_parameters(network, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(network.weights, f)
        pickle.dump(network.biases, f)

if __name__ == '__main__':
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    main_dir = "train_data/train"  # 这里应该是数据存放的主目录路径
    # 遍历主目录下的子目录
    for subdir_name in sorted(os.listdir(main_dir)):
        subdir_path = os.path.join(main_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue  # 如果不是子目录，则跳过

        # 为每个子目录分配一个唯一的独热编码
        label = int(subdir_name) - 1  # 子目录名称从1开始，独热编码从0开始
        one_hot_label = np.eye(12)[label]

        # 遍历子目录中的图像文件
        image_files = os.listdir(subdir_path)
        np.random.shuffle(image_files)  # 洗牌功能

        for i, filename in enumerate(image_files):
            # 读取图像
            image_path = os.path.join(subdir_path, filename)
            img = Image.open(image_path)
            # 将图像转换为灰度图像
            gray_img = img.convert('L')
            # 将灰度图像转换为NumPy数组
            img_array = np.array(gray_img)
            # 将图像数据添加到训练或测试数据中
            normalized_img = img_array / 255.0
            if i < len(image_files) * 0.8:  # 80%的数据用于训练
                train_inputs.append(normalized_img.reshape(1, -1))
                train_labels.append(one_hot_label)
            else:  # 其余20%的数据用于测试
                test_inputs.append(normalized_img.reshape(1, -1))
                test_labels.append(one_hot_label)

    # 将列表转换为NumPy数组
    X_train = np.vstack(train_inputs)
    y_train = np.vstack(train_labels)
    X_test = np.vstack(test_inputs)
    y_test = np.vstack(test_labels)

    # 创建神经网络实例
    input_size = 28 * 28  # 输入大小
    hidden_layers = 2# 隐藏层数量
    neurons_per_layer = [128,64]  # 每个隐藏层的神经元数量
    output_size = 12  # 输出大小
    dropout_rate= 0.2
    nn = NeuralNetwork(input_size, hidden_layers, neurons_per_layer, output_size,1-dropout_rate)

    # 定义训练超参数
    learning_rate = 0.00007
    epochs = 4000

    nn.train(X_train, y_train, learning_rate, epochs)
    # 使用测试数据进行评估
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = len(test_inputs)

    accuracy = evaluate_model(nn, X_test, y_test)
    print(f"Test Accuracy: {accuracy:.7f}")
    # 保存训练好的模型参数
    save_model_parameters(nn, 'trained_model_parameters.pkl')

