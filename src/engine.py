import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train(model, dataloader, optimizer, criterion, train_data, device):
    print('训练中...')
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_predictions = []
    train_targets = []

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        predicted_labels = (outputs > 0.5).float()
        train_predictions.append(predicted_labels.detach().cpu())
        train_targets.append(target.detach().cpu())

    train_loss = train_running_loss / counter
    train_predictions = torch.cat(train_predictions, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_accuracy = accuracy_score(train_targets, train_predictions)
    train_precision = precision_score(train_targets, train_predictions, average='micro')
    train_recall = recall_score(train_targets, train_predictions, average='micro')

    return train_loss, train_accuracy, train_precision, train_recall


def validate(model, dataloader, criterion, val_data, device):
    print('验证中...')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

            predicted_labels = (outputs > 0.5).float()
            val_predictions.append(predicted_labels.detach().cpu())
            val_targets.append(target.detach().cpu())

        val_loss = val_running_loss / counter
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_precision = precision_score(val_targets, val_predictions, average='micro')
        val_recall = recall_score(val_targets, val_predictions, average='micro')

    return val_loss, val_accuracy, val_precision, val_recall
