from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import utils
import ipdb


class MovingMNIST(Dataset):
    def __init__(self, filename, inp_len=10, out_len=1, transforms=None):
        # moving mnist dataset 

        # transform uint8 image to float
        self.data = np.load(filename).astype(np.float)/255.0
        # (seq, num_examples, num_channels=1, size, size)
        self.data = np.expand_dims(self.data, axis=2)
        self.inp_len = inp_len
        self.out_len = out_len
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        inp = self.data[:self.inp_len, index, :, :]
        out = self.data[self.inp_len:self.inp_len+self.out_len, index, :, :]
        if self.transforms is not None:
            inp = self.transforms(inp)
            out = self.transforms(out)

        return inp, out


class STDataset(Dataset):
    def __init__(self, data, params):
        super(STDataset, self).__init__()
        self.seq_length = params['seq_length']
        self.future = params['future']
        self.random_roi = params['random_roi']
        self.roi_size = params['roi_size']

        if not self.random_roi:
            data = data[:, params['lat_min']:params['lat_max'], params['lon_min']:params['lon_max']]

        # transforming data to (time, channels=1, height, width)
        self.data = np.expand_dims(data, 1)
        self.size = len(self.data)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.data[idx: idx+self.seq_length]
        target = self.data[idx+self.seq_length: idx+self.seq_length+1+self.future]
        if self.random_roi:
            r1,r2,c1,c2 = utils.get_random_roi(*self.data.shape[-2:], self.roi_size)
            inp = inp[:, :, r1:r2, c1:c2]
            target = target[:, :, r1:r2, c1:c2]
        return inp, target 


def get_st_dataloaders(args):
    # return train and test dataloaders for conv model
    air_data = utils.load_nc_data(args.data_file)
    # normalize data between 0 and 1
    if args.normalize_y:
        air_data = (air_data - air_data.min())/(air_data.max() - air_data.min())
    data_mean = air_data.mean(0)
    air_data = air_data - data_mean

    # train test split
    train_indices = np.arange(600)
    test_indices = np.arange(636, 800)
    eval_data = air_data[636:800]
    if not args.random_roi:
        eval_data = eval_data[:, args.lat_min:args.lat_max, args.lon_min:args.lon_max]
        data_mean = data_mean[args.lat_min:args.lat_max, args.lon_min:args.lon_max]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_batch_size = args.batch_size
    test_batch_size = args.batch_size

    dataset = STDataset(data=air_data, params=vars(args))
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=0, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, num_workers=0, sampler=test_sampler)
    return train_dataloader, test_dataloader, data_mean


def get_mnist_dataloader(args):
    filename = 'data/mnist_test_seq.npy'
    inp_len = args.seq_length
    out_len = 1
    dataset = MovingMNIST(filename, inp_len=inp_len, out_len=out_len)
    num_train = int(.75*len(dataset))

    train_indices = np.arange(num_train)
    test_indices = np.arange(num_train, len(dataset))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_batch_size = args.batch_size
    test_batch_size = args.batch_size
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=0, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, num_workers=0, sampler=test_sampler)
    
    # sz = train_dataloader.dataset.data.shape[-2:]
    # data_mean = train_dataloader.dataset.data.reshape(-1,sz[0],sz[1]).mean(0)
    data_mean = None
    return train_dataloader, test_dataloader, data_mean