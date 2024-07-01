import os
import random
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from CNN import CustomDataset, CNN
if __name__ == '__main__':
    # 加载保存的模型
    model = CNN()
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # 将模型设置为评估模式，这会关闭 Dropout 等模型中的随机操作
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转换为灰度图像
        transforms.ToTensor()    # 转换为Tensor
    ])

    # 数据集根目录
    root_dir = 'train_data/test_data'

    # 构建数据集
    all_data = datasets.ImageFolder(root=root_dir, transform=transform)

    # 获取所有类别及其对应的文件夹路径
    class_folders = all_data.class_to_idx.items()
    # 划分训练集和测试集

    test_data = []

    for class_name, class_folder in class_folders:
        class_path = os.path.join(root_dir, class_name)
        images = [img_name for img_name in os.listdir(class_path) if img_name.endswith('.bmp')]
        random.shuffle(images)

        num_images = len(images)


        # 将数据划分为训练集和测试集
        for i, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            img_data = (img_path, class_folder)
            test_data.append(img_data)


    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_data, transform=transform),
        batch_size=32, shuffle=False, num_workers=2
    )
    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %.5f' % (
            100 * correct / total))
