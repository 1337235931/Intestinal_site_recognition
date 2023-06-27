import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]['image_file']
        self.all_labels = np.array(self.csv.drop(['image_file'], axis=1))
        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio

        # 设置训练数据集的图像和标签
        if self.train:
            print(f"训练集图像数量: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            # 定义训练数据的转换操作
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((640, 360)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])

        # 设置验证数据集的图像和标签
        elif not self.train and not self.test:
            print(f"验证集图像数量: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-500])
            self.labels = list(self.all_labels[-self.valid_ratio:-500])
            # 定义验证数据的转换操作
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((640, 360)),
                transforms.ToTensor(),
            ])

        # 设置测试数据集的图像和标签，只使用最后10张图像
        # 在单独的推断脚本中使用
        elif self.test and not self.train:
            self.image_names = list(self.all_image_names[-500:])
            self.labels = list(self.all_labels[-500:])
            # 定义测试数据的转换操作
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(f"../input/movie-classifier/Multi_Label_dataset/Images/{self.image_names[index]}")
        # 将图像从BGR格式转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 应用图像转换操作
        image = self.transform(image)
        targets = self.labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
