import numpy as np
import pickle
import os
from PIL import Image
from BP import NeuralNetwork, evaluate_model
def load_model_parameters(filepath):
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
        biases = pickle.load(f)
    return weights, biases
if __name__ == '__main__':
    # 加载训练好的模型参数
    loaded_weights, loaded_biases = load_model_parameters('trained_model_parameters.pkl')

    # 创建神经网络实例并加载参数
    input_size = 28 * 28
    hidden_layers = 1
    neurons_per_layer = [256]
    output_size = 12
    dropout_rate = 0.2
    nn = NeuralNetwork(input_size, hidden_layers, neurons_per_layer, output_size, dropout_rate)
    nn.weights = loaded_weights
    nn.biases = loaded_biases


    test_inputs = []
    test_labels = []
    main_dir = "train_data/test_data"  # 这里应该是数据存放的主目录路径
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
            test_inputs.append(normalized_img.reshape(1, -1))
            test_labels.append(one_hot_label)
    # 将列表转换为NumPy数组
    X_test = np.vstack(test_inputs)
    y_test = np.vstack(test_labels)
    accuracy = evaluate_model(nn, X_test, y_test)
    print(f"Test Accuracy: {accuracy:.7f}")