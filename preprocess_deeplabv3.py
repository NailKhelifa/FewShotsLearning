import numpy as np
import pandas as pd
import torch
import os
import cv2
from pathlib import Path
data_dir = os.getcwd() + '/data'


def load_dataset(dataset_dir):
    dataset_list = []
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)


data_train = load_dataset(data_dir + "/X_train")
data_test = load_dataset(data_dir + "/X_test")

labels_train = pd.read_csv(data_dir + "/Y_train.csv", index_col=0).T


def consecutive_values(row):
    """Modifie les valeurs de la ligne pour obtenir des entiers entre 0 et le nb de classes sur l'image tout en conservant les différences"""
    l, _ = pd.factorize(row, sort=True)
    return l


labels_train = labels_train.to_numpy()
labels_trainr = np.array([consecutive_values(row) for row in labels_train])
labels_trainr = pd.DataFrame(labels_trainr)


labels = []
for k in range(len(labels_trainr)) :
    labels.append(torch.tensor(np.array(labels_trainr.iloc[k]).reshape(512, 512)))


y_train = torch.stack(labels[0:400])
x_train = torch.tensor(data_train[0:400]).unsqueeze(1)  # unsqueeze pour la dimension des channels de couleur (1 car greyscale)

y_valid = torch.stack(labels[800:])
x_valid = torch.tensor(data_train[800:]).unsqueeze(1)

x_test = torch.tensor(data_test).unsqueeze(1)


torch.save(y_train, data_dir+"/y_train.pt")
torch.save(x_train, data_dir+"/x_train.pt")
torch.save(y_valid, data_dir+"/y_valid.pt")
torch.save(x_valid, data_dir+"/x_valid.pt")
torch.save(x_test, data_dir+"/x_test.pt")



