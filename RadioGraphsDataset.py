from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
from utils import parse_data
import pandas as pd
import pydicom as pdc
import torch
import numpy as np
import sys
import time
import os
import random

class RadioGraphsDataset(Dataset):
    def __init__(self, training, args):
        self.training = training
        self.classification = True
        self.augment = False
        self.data_dir = 'D:/all/'
        self.df = pd.read_csv(os.path.join(self.data_dir,'stage_1_train_labels.csv'))
        self.truth = parse_data(self.df)

        if self.training:
            self.data_index, _, _,_ = train_test_split(np.arange(len(self.truth)), np.arange(len(self.truth)), test_size=0.25, random_state=42)


        else:
            _, self.data_index, _,_ = train_test_split(np.arange(len(self.truth)), np.arange(len(self.truth)), test_size=0.25, random_state=42)
            self.length = len(self.data_index)

        self.length = len(self.data_index)
        with open('val.txt','w') as f:
            for i in range(self.length):
                f.write(self.df['patientId'][self.data_index[i]] +'\n')



    def __len__(self):
        return self.length

    def __getitem__(self, index):
        patientId = self.df['patientId'][self.data_index[index]]
        dcm_file = self.data_dir + '/stage_1_train_images/%s.dcm' % patientId
        dcm_data = pdc.read_file(dcm_file).pixel_array
        dcm_data = np.stack([dcm_data] * 3, axis=2)
        if self.classification:
            data = cv2.resize(dcm_data,(224,224))
            data = data.transpose(2,0,1)
            return torch.tensor(data), torch.tensor(self.truth[patientId]['present'])
        else:
            # todo
            bbox = self.truth[patientId]['boxes']
            return torch.tensor(data), torch.tensor(bbox)

if __name__ == "__main__":
    dataset = RadioGraphsDataset(training = False, args="hi")
    data, truth =  dataset.__getitem__(4)
    print(data.shape)
