import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss
from models import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model(pretrained=False, requires_grad=False).to(device)

checkpoint = torch.load('../outputs/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_csv = pd.read_csv('../input/movie-classifier/Multi_Label_dataset/train.csv')
genres = train_csv.columns.values[1:]

test_data = ImageDataset(train_csv, train=False, test=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

accuracy = 0.0
precision = 0.0
recall = 0.0
subset_accuracy = 0.0
total_samples = 0

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    target_cls = len(target_indices)
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    sorted_cls = target_cls
    best = sorted_indices[-sorted_cls:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{genres[best[i]]}\n"
    for i in range(len(target_indices)):
        string_actual += f"{genres[target_indices[i]]}\n"

    predicted_indices = [best[i] for i in range(len(best))]
    actual_indices = target_indices
    accuracy += accuracy_score(actual_indices, predicted_indices)
    precision += precision_score(actual_indices, predicted_indices, average='micro')
    recall += recall_score(actual_indices, predicted_indices, average='micro')
    subset_accuracy += hamming_loss(actual_indices, predicted_indices)
    total_samples += 1

    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    plt.imshow(image)
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.text(
        0, 0.5, f"PREDICTED:\n{string_predicted}",
        verticalalignment='center', horizontalalignment='left',
        transform=plt.gca().transAxes, color='white', fontsize=12
    )
    plt.text(
        0, 0.8, f"ACTUAL:\n{string_actual}",
        verticalalignment='center', horizontalalignment='left',
        transform=plt.gca().transAxes, color='white', fontsize=12
    )
    plt.savefig(f"../outputs/inference_{counter}.jpg", dpi=300, quality=95)
    plt.show()

accuracy /= total_samples
precision /= total_samples
recall /= total_samples
subset_accuracy /= total_samples

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"子集准确率: {subset_accuracy:.4f}")
