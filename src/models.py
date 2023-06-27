from torchvision import models as models
import torch.nn as nn


# 创建模型函数
def create_model(pretrained, requires_grad):
    model = models.resnet101(progress=True, pretrained=pretrained)
    # 冻结隐藏层
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    # 训练隐藏层
    elif requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    # 设置分类层为可学习
    # 总共有10个类别
    model.fc = nn.Linear(2048, 8)
    return model
