import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from models import create_model
from dataset import ImageDataset
from engine import train, validate
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model(pretrained=False, requires_grad=False).to(device)

train_csv = pd.read_csv('../input/movie-classifier/Multi_Label_dataset/train.csv')
train_data = ImageDataset(train_csv, train=True, test=False)
val_data = ImageDataset(train_csv, train=False, test=False)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

criterion = nn.BCELoss()  # 使用适当的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []

best_val_loss = float('inf')

epochs = 100

for epoch in range(epochs):
    print(f"第 {epoch + 1} 个Epoch，总共 {epochs} 个Epoch")

    train_loss, train_accuracy, train_precision, train_recall = train(model, train_loader, optimizer, criterion, train_data, device)
    val_loss, val_accuracy, val_precision, val_recall = validate(model, val_loader, criterion, val_data, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 将训练好的模型保存到磁盘
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, '../outputs/model.pth')

    print(
        f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}\n'
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}\n'
    )

# 绘制训练和验证指标图表
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('../outputs/loss_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('../outputs/accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_precisions, label='Train Precision')
plt.plot(val_precisions, label='Val Precision')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Training and Validation Precision')
plt.savefig('../outputs/precision_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_recalls, label='Train Recall')
plt.plot(val_recalls, label='Val Recall')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Training and Validation Recall')
plt.savefig('../outputs/recall_plot.png')
plt.show()
