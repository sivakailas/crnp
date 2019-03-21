import numpy as np
import utils
from arguments import get_args
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tensorboard_logger
from train import Trainer


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


def get_dataloaders(args):
    # return train and test dataloaders for conv model
    air_data = utils.load_nc_data(args.data_file)
    # normalize data between 0 and 1
    if args.normalize_y:
        air_data = (air_data - air_data.min())/(air_data.max() - air_data.min())
    data_mean = air_data.mean(0)
    air_data = air_data - data_mean

    # train test split
    train_indices = np.arange(50, 600)
    test_indices = np.arange(636, 800)
    eval_data = air_data[636:800]
    if not args.random_roi:
        eval_data = eval_data[:, args.lat_min:args.lat_max, args.lon_min:args.lon_max]
        data_mean = data_mean[args.lat_min:args.lat_max, args.lon_min:args.lon_max]

    shuffle = False
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_batch_size = 64
    test_batch_size = 64

    dataset = STDataset(data=air_data, params=vars(args))
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=1, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, num_workers=1, sampler=test_sampler)
    return train_dataloader, test_dataloader, eval_data, data_mean


if __name__ == '__main__':
    args = get_args()
    if not args.test:
        tensorboard_logger.configure(args.logdir)
        print('Logging to {}'.format(args.logdir))

    # first variant - for every timeframe, select k points 
    # later work - work with different k 
    
    train_dataloader, test_dataloader, eval_data, data_mean = get_dataloaders(args)
    trainer = Trainer(train_dataloader, test_dataloader, args=args, eval_data=eval_data, data_mean=data_mean)
    trainer.train(args.num_epochs)