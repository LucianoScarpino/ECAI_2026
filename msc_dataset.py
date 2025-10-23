from torch.utils.data import Dataset

import torch
import torchaudio
import os

class MSCDataset(Dataset):
    """
    INPUT:
    - data_path : str -> path containing audio files in .wav format
    - classes : list[str] -> list containing ordered class labels

    """
    def __init__(self, data_path: str, classes: list[str]):
        self.data = self.unpack_audios_(data_path)
        self.encoded_labels = self.encode_labels_(classes)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        waveform, sampling_rate, label = self.data[idx]
        return {
            'x': torch.tensor(waveform,dtype=torch.float32),
            'sampling_rate': sampling_rate,
            'label': label
        }
    
    def unpack_audios_(self,path):
        audios = []
        for file_name in os.listdir(path):
            if file_name.endswith(".wav"):
                label = file_name.split("_")[0]
                file_path = os.path.join(path,file_name)
                waveform,sample_rate = torchaudio.load(file_path)
                audios.append((waveform,sample_rate,label))

        return audios
    
    def encode_labels_(self,labels):
        encoded = {}
        for idx,label in enumerate(labels):
            encoded[label] = idx

        return encoded
    
    def classes(self,label):
        return self.encoded_labels[label]
    
    def label_to_int(self,label):
        if label in self.encoded_labels.keys():
            return self.encoded_labels[label]