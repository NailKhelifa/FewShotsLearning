from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import time


class MulticlassDiceLoss(nn.Module):
    """ Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets, smooth=1e-6):
        """Computes the dice loss for all classes and provides an overall weighted loss."""
        probabilities = logits

        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=self.num_classes)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()

        mod_a = intersection.sum()
        mod_b = targets.numel()

        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss


def postprocess(batch):
  return F.interpolate(batch, 512, mode = 'nearest-exact')


class DataSet_with_transform(Dataset):
    def __init__(self, x_dataset, y_dataset, transform=None):
        self.x = x_dataset
        self.y = y_dataset
        assert len(self.x) == len(self.y), "x and y should have same length"
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return [x, y]

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.225])])

def postprocess(batch):
  return F.interpolate(batch, 512, mode = 'nearest-exact')


if __name__ == "__main__":
    device = torch.device("cuda")
    data_dir = 'data'
    num_epochs = 100
    batch_size = 16
    lr = 0.005
    num_classes = 31 # le nombre de classes à détecter, 0 inclus
    in_channels = 256 # le nombre de canaux en entrée du classifier

    # resize = 520 # le resize adapté pour le modèle
    # resize = 512



    weights = DeepLabV3_ResNet50_Weights.DEFAULT # on commence par charger les weights
    model = deeplabv3_resnet50(weights=weights) # instanciation du modèle
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # on modifie la première couche pour prendre du greyscale


    y_train = torch.load(data_dir+'/y_train.pt')
    x_train = torch.load(data_dir+'/x_train.pt')
    y_valid = torch.load(data_dir+'/y_valid.pt')
    x_valid = torch.load(data_dir+'/x_valid.pt')

    train_dataset = DataSet_with_transform(x_train, y_train, transform = preprocess)
    valid_dataset = DataSet_with_transform(x_valid, y_valid, transform = preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True, pin_memory = True)

    model.classifier[4] = DeepLabHead(in_channels, num_classes) # on change le classifier pour avoir le bon nombre de classes

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    loss_over_epochs = []
    criterion = MulticlassDiceLoss(num_classes=num_classes)

    for epoch in range(num_epochs):
        batch_loss = []
        print(epoch)
        epoch_start_time = time.time()  # Mesure du temps de l'époque
        for x, y in train_dataloader:
            start_time = time.time()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = postprocess(model(x)['out'].softmax(dim=1))        
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            
            batch_time = time.time() - start_time  
            torch.cuda.empty_cache()    
        epoch_time = time.time() - epoch_start_time  # Calcul du temps écoulé pour l'époque
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {np.mean(batch_loss)}, Mean batch Time: {epoch_time/100:.4f} seconds') 
        
        loss_over_epochs.append(np.mean(batch_loss))

    torch.save(model.state_dict(), 'results/deeplabv3resnet.pt')
    torch.save(loss_over_epochs, 'results/loss_deeplabv3resnet.pt')

