import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class meta_img_dataset(Dataset):
    def __init__(self, img_path, meta_data, labels, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.labels = labels
        self.meta_data = meta_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        label = self.labels[index]
        meta_data = self.meta_data[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(meta_data).float(), label

class meta_img_dataset_test(Dataset):
    def __init__(self, img_path, meta_data, labels, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.labels = labels
        self.meta_data = meta_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        label = self.labels[index]
        meta_data = self.meta_data[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(meta_data).float(), label

meta_data_columns = [
    'smoke_False', 'smoke_True', 'drink_False', 'drink_True', 'background_father_POMERANIA',
    'background_father_GERMANY', 'background_father_BRAZIL', 'background_father_NETHERLANDS',
    'background_father_ITALY', 'background_father_POLAND', 'background_father_UNK',
    'background_father_PORTUGAL', 'background_father_BRASIL', 'background_father_CZECH',
    'background_father_AUSTRIA', 'background_father_SPAIN', 'background_father_ISRAEL',
    'background_mother_POMERANIA', 'background_mother_ITALY', 'background_mother_GERMANY',
    'background_mother_BRAZIL', 'background_mother_UNK', 'background_mother_POLAND',
    'background_mother_NORWAY', 'background_mother_PORTUGAL', 'background_mother_NETHERLANDS',
    'background_mother_FRANCE', 'background_mother_SPAIN', 'age', 'pesticide_False',
    'pesticide_True', 'gender_FEMALE', 'gender_MALE', 'skin_cancer_history_True',
    'skin_cancer_history_False', 'cancer_history_True', 'cancer_history_False',
    'has_piped_water_True', 'has_piped_water_False', 'has_sewage_system_True',
    'has_sewage_system_False', 'fitspatrick_3.0', 'fitspatrick_1.0', 'fitspatrick_2.0',
    'fitspatrick_4.0', 'fitspatrick_5.0', 'fitspatrick_6.0', 'region_ARM', 'region_NECK',
    'region_FACE', 'region_HAND', 'region_FOREARM', 'region_CHEST', 'region_NOSE', 'region_THIGH',
    'region_SCALP', 'region_EAR', 'region_BACK', 'region_FOOT', 'region_ABDOMEN', 'region_LIP',
    'diameter_1', 'diameter_2', 'itch_False', 'itch_True', 'itch_UNK', 'grew_False', 'grew_True',
    'grew_UNK', 'hurt_False', 'hurt_True', 'hurt_UNK', 'changed_False', 'changed_True',
    'changed_UNK', 'bleed_False', 'bleed_True', 'bleed_UNK', 'elevation_False', 'elevation_True',
    'elevation_UNK'
]