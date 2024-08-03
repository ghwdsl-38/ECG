from ECGTrain import ECGTrain,fix_random
import torch

train_dataset = torch.load("./data/mit/train.pt")
val_dataset = torch.load("./data/mit/val.pt")
test_dataset = torch.load("./data/mit/test.pt")
ecg_train = ECGTrain(train_dataset, val_dataset, test_dataset)
ecg_train.train()