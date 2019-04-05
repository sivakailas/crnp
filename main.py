from arguments import get_args
import tensorboard_logger
from train import Trainer
from dataloaders import get_mnist_dataloader, get_st_dataloaders


if __name__ == '__main__':
    args = get_args()
    if not args.test:
        tensorboard_logger.configure(args.logdir)
        print('Logging to {}'.format(args.logdir))

    # first variant - for every timeframe, select k points 
    # later work - work with different k 
    
    if args.mnist:
        train_dataloader, test_dataloader, data_mean = get_mnist_dataloader(args)
    else:
        train_dataloader, test_dataloader, data_mean = get_st_dataloaders(args)
    trainer = Trainer(train_dataloader, test_dataloader, args=args, data_mean=data_mean)
    trainer.train(args.num_epochs)