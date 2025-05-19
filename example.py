import pandas as pd 


data1 = pd.read_csv("data1.csv")



data2 = pd.read_csv("data1.csv")


data = pd.concat([data1, data2], axis=0, ignore_index=True)

print(data1.info())
print(100*"#")
print(data.info())





# from torch.utils.data import Dataset, DataLoader

# import numpy as np
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Assuming the last column is the label
#         x = self.data.iloc[idx, :-1].values.astype(np.float32)
#         y = self.data.iloc[idx, -1]
#         return x, y


