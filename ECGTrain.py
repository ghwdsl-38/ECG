import torch
from ECGDataset import ECGDataset
from ECGConfig import MIT_BIH
from ECGLoss import ECGLoss
from ECGModel import ECGModel
from collections import Counter
from torch.utils.data import DataLoader
import numpy as np 
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import random
import os

def fix_random(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ECGTrain:
    def __init__(self, train_dataset, val_dataset, test_dataset):
        # Load datasets
        self.train_dataset = ECGDataset(train_dataset)
        self.val_dataset = ECGDataset(val_dataset)
        self.test_dataset = ECGDataset(test_dataset)
        self.best_macro_f1 = 0.0
        self.model = None
        self.train_data = []
        self.val_data = []

    def train(self):
        fix_random(0)
        
        batch_size = 128
        train_loader = DataLoader(dataset=self.train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True, 
                                  drop_last=True, 
                                  num_workers=0)
        val_loader = DataLoader(dataset=self.val_dataset, 
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=True, 
                                num_workers=0)
        test_loader = DataLoader(dataset=self.test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False, 
                                 drop_last=False, 
                                 num_workers=0)
        labels = self.train_dataset.labels.numpy().tolist()

        # Count the occurrences of each class
        counter = Counter(labels)

        # Convert to dictionary with keys as class indices
        class_dict = {i: counter[i] for i in range(len(np.unique(labels)))}

        criterion = ECGLoss(class_dict).criterion()

        mit = MIT_BIH()

        self.model = ECGModel(mit.num_classes)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            betas=(0.9, 0.99)
        )

        for epoch in range(1, 50 + 1):
            self.model.train()
            print(f'[Epoch : {epoch}/50]')
            for step, batches in enumerate(train_loader):
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # ====== Source =====================
                optimizer.zero_grad()

                # Src original features
                logits = self.model(data)

                # Cross-Entropy loss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

            # Evaluate after each epoch
            train_metrics = self.evaluate(self.model, train_loader, criterion)
            val_metrics = self.evaluate(self.model, val_loader, criterion)
            self.train_data.append(train_metrics)
            self.val_data.append(val_metrics)

            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Macro F1: {val_metrics['macro_f1']:.4f}")

            if val_metrics['macro_f1'] > self.best_macro_f1:
                self.best_macro_f1 = val_metrics['macro_f1']
                self.save_model(val_metrics, self.model)
        
        
        # Evaluate on the test dataset using the best model
        self.model.load_state_dict(torch.load('./model/checkpoint.pt'))
        test_metrics = self.evaluate(self.model, test_loader, criterion)
        print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, Test Macro F1: {test_metrics['macro_f1']:.4f}")

        # Plot accuracy and loss
        self.plot_metrics()

    def evaluate(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batches in dataloader:
                data = batches['samples'].float()
                labels = batches['labels'].long()

                logits = model(data)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        macro_f1 = np.mean(f1)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1
        }

        return metrics

    def save_model(self, metrics, model):
        save_dict = {
            "dataset": self.train_dataset,
            "model": model.state_dict(),
            "clf": model.clf.state_dict(),
            "metrics": metrics
        }
        save_path = "./model/checkpoint.pt"
        torch.save(save_dict, save_path)
    
    def plot_metrics(self):
        epochs = range(1, 51)
        train_loss = [data['loss'] for data in self.train_data]
        val_loss = [data['loss'] for data in self.val_data]
        train_acc = [data['accuracy'] for data in self.train_data]
        val_acc = [data['accuracy'] for data in self.val_data]

        # Plot loss
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


