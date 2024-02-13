import torch
from torch.utils.data import Dataset
import pandas as pd


class PlantDataset(Dataset):
    def __init__(self, is_train= True):
        self.is_train = 0
        if is_train:
            self.is_train = 1
        df = pd.read_csv("data/info.csv")
        df = df[df["is_train"] == self.is_train]

        self.label_mapping = {"S": 0, "T": 1}
        self.group = []
        for group in list(df["Group"]):
            self.group.append(self.label_mapping[group])
        self.group = torch.tensor(self.group)
        columns_to_convert = ['Genotype', 'Treatment', 'Replication']
        df = pd.get_dummies(df, columns=columns_to_convert)
        columns_to_drop = ['File.Name', 'Group', 'Day']
        df = df.drop(columns=columns_to_drop, axis=1)
        df = df.replace({False: 0, True: 1})
        self.data = torch.tensor(df.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.group)

    def __getitem__(self, idx):
        group = self.group[idx]
        data = self.data[idx]
        return data, group


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = PlantDataset(is_train=True)
    dl = DataLoader(ds, batch_size=1000, shuffle=True)
    for data, group in dl:
        print(data.shape)
        print(group.shape)
        break