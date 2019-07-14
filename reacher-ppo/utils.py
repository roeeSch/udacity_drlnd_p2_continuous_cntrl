from torch.utils import data as tData


class Dataset(tData.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataLen):
        'Initialization'
        self.indxs = list(range(dataLen))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.indxs)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.indxs[index]


class miniBatcher:
    def __init__(self,  dataLength, batch_size=32, shuffle=True, num_workers=1):

        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}

        self.indices = Dataset(dataLength)

        self.generator = tData.DataLoader(self.indices, **params)

