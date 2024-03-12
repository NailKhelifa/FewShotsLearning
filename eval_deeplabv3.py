import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
from deeplabv3resnet import DataSet_with_transform, preprocess, postprocess, MulticlassDiceLoss
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import adjusted_rand_score
import numpy as np


data_dir = 'data'
y_valid = torch.load(data_dir+'/y_valid.pt')
x_valid = torch.load(data_dir+'/x_valid.pt')

num_classes = 31
in_channels = 256
batch_size = 8
device = torch.device("cuda")


#Load the model

model = deeplabv3_resnet50()
model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # on modifie la première couche pour prendre du greyscale
model.classifier[4] = DeepLabHead(in_channels, num_classes) 
model.load_state_dict(torch.load('./results/deeplabv3resnet.pt'))


#Load data

y_valid = torch.load(data_dir+'/y_valid.pt')
x_valid = torch.load(data_dir+'/x_valid.pt')
valid_dataset = DataSet_with_transform(x_valid, y_valid, transform = preprocess)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, pin_memory = True)


#Define accuracy function

def rand_index(y_true, y_pred) -> float: 
    """Compute the individual rand index for each sample in the dataset and then average it"""
    individual_rand_index = []
    for img_index in range(y_true.shape[0]):
        labels = y_true[img_index, :, :]
        preds = y_pred[img_index, :, :]
        individual_rand_index.append(adjusted_rand_score(labels, preds))
    return np.mean(individual_rand_index)


#Make predictions

loss_f = MulticlassDiceLoss(num_classes=num_classes)
model.eval()
predictions = []
batch_loss = []
accuracies = []
for x, y in valid_dataloader:
    x, y = x.to(device), y.to(device)
    outputs = postprocess(model(x)['out'].softmax(dim=1))   

    loss = loss_f(outputs, y)
    preds = outputs.argmax(axis = 1)
    acc = rand_index(y, preds)

    batch_loss.append(loss)
    accuracies.append(acc)
    predictions.append(preds)


#Results

mean_loss = batch_loss.mean()
mean_accuracy = accuracies.mean()


print("mean loss: ", mean_loss)
print("mean accuracy: ", mean_accuracy)

torch.save(preds, "results/preds.pt")