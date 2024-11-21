import os
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,roc_auc_score,precision_recall_curve,recall_score, auc
import matplotlib.pyplot as plt
from swin_transformer2_HS2 import SwinTransformer
import torch.nn.functional as F
import torch.nn as nn



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Configs :
device = torch.device("cuda")
save_interval = 10
# model save dir
save_path = './modelA2/pres01'
# tensorboard log dir
log_dir = './modelA2/log2/'
# dataset path: directories containing jpg or png images
batch_size = 32
data_dir = 'F:/coffee/jyj/swintransformer/coffee_aajpd2'
input_size = 224

class CombinedFocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=12, weight_ce=0.7, weight_focal=0.3):
        super(CombinedFocalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_focal = weight_focal

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets)

        # 计算 Focal Loss
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        BCE_loss = BCE_loss.view(-1, 1)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        focal_loss = (focal_loss * targets_one_hot).sum(dim=1).mean()

        # 结合交叉熵损失和 Focal Loss
        combined_loss = self.weight_ce * ce_loss + self.weight_focal * focal_loss
        return combined_loss


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.CenterCrop((input_size, input_size)),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
    for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=8)
    for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
print(image_datasets['train'].class_to_idx)

def get_classification_metrics(true_labels, predicted_labels, num_classes):
    metrics = {
        'TP': np.zeros(num_classes, dtype=int),
        'FP': np.zeros(num_classes, dtype=int),
        'TN': np.zeros(num_classes, dtype=int),
        'FN': np.zeros(num_classes, dtype=int),
        'Precision': np.zeros(num_classes, dtype=float),  # 准确率
        'Recall': np.zeros(num_classes, dtype=float),     # 召回率
    }
    
    for c in range(num_classes):
        metrics['TP'][c] = np.sum((predicted_labels == c) & (true_labels == c))
        metrics['FP'][c] = np.sum((predicted_labels == c) & (true_labels != c))
        metrics['TN'][c] = np.sum((predicted_labels != c) & (true_labels != c))
        metrics['FN'][c] = np.sum((predicted_labels != c) & (true_labels == c))

        # Calculate Precision for class c
        metrics['Precision'][c] = metrics['TP'][c] / (metrics['TP'][c] + metrics['FP'][c]) if (metrics['TP'][c] + metrics['FP'][c]) != 0 else 0

        # Calculate Recall (or Sensitivity) for class c
        metrics['Recall'][c] = metrics['TP'][c] / (metrics['TP'][c] + metrics['FN'][c]) if (metrics['TP'][c] + metrics['FN'][c]) != 0 else 0
        
    return metrics


def plot_to_tensorboard(cm, class_names):
    fig = plt.figure(figsize=(50, 50))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.close(fig)
    return fig

def train_model(model, criterion, optimizer, start_epoch=0, num_epochs=20, scheduler=None):
    class_names = ["01AA", "02AA", "03JPAA", "04ganguo", "05ganguosuipian", "06heidou", "07meidou", "08posuidou", "09suandou", "10waike", "11weichengshudou", "12yangpizhidou"]
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, num_epochs):
        if epoch % save_interval == 0:
            print(f'Save model:Epoch {epoch}')
            torch.save(model, os.path.join(save_path, f'{epoch}.pth'))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_preds = []
            running_labels = []
            running_scores = []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    scores = torch.nn.functional.softmax(outputs, dim=1) # 记录预测得分
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_preds.append(preds)
                running_labels.append(labels)
                running_scores.append(scores)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_preds = torch.cat(running_preds).cpu().numpy()
            epoch_labels = torch.cat(running_labels).cpu().detach().numpy()
            epoch_scores = torch.cat(running_scores).cpu().detach().numpy()
            
            acc = accuracy_score(epoch_labels, epoch_preds)
            f1 = f1_score(epoch_labels, epoch_preds, average='weighted')

            writer.add_scalar(f'Acc/{phase}', acc, epoch)
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'F1/{phase}', f1, epoch)
            class_metrics = get_classification_metrics(epoch_labels, epoch_preds, len(class_names))
            for c, class_name in enumerate(class_names):
                writer.add_scalar(f'Precision/{class_name}/{phase}', class_metrics['Precision'][c], epoch)
                writer.add_scalar(f'Recall/{class_name}/{phase}', class_metrics['Recall'][c], epoch)
            
           
            APs = []
            for c in range(len(class_names)):
                y_true_class = (epoch_labels == c).astype(int)
                y_scores_class = epoch_scores[:, c]
                precision, recall, _ = precision_recall_curve(y_true_class, y_scores_class)
                AP = auc(recall, precision)
                APs.append(AP)

            mAP = np.mean(APs)
            writer.add_scalar(f'AP/{phase}', np.mean(APs), epoch)
            writer.add_scalar(f'mAP/{phase}', mAP, epoch)

            if phase == 'val':
                cm = confusion_matrix(epoch_labels, epoch_preds)
                for i, row in enumerate(cm):
                    for j, count in enumerate(row):
                        if i != j and count > 0:
                            print(f"{count} samples of class {class_names[i]} were predicted as {class_names[j]}")

                fig = plot_to_tensorboard(cm, class_names)
                writer.add_figure('Confusion Matrix', fig, global_step=epoch)

    return model



if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    model = SwinTransformer(num_classes=12)
    # Train the model for 50 epochs without pre-trained weights
    criterion = CombinedFocalCrossEntropyLoss(alpha=0.25, gamma=2.0, num_classes=12, weight_ce=0.5, weight_focal=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_model(model, criterion, optimizer, num_epochs=50, start_epoch=0, scheduler=scheduler)

    # Save the trained model after 50 epochs
    torch.save(model.state_dict(), './modelA2/model_sw_newdata2/50.pth')