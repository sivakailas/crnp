import os
import argparse
import datetime


def get_args():
    parser = argparse.ArgumentParser()

    # DATASET
    # data_file = 'noaa_datasets/air.sig995.mon.mean.nc'
    data_file = 'animal_data.mat'
    parser.add_argument('--data-file', default=data_file)
    parser.add_argument('--random-roi', action='store_true', help='A random region of interest is sampled at every instant')
    parser.add_argument('--roi-size', default=30, type=int, help='size of region of interest')
    # these are indices, check the data file to see the actual values of them
    parser.add_argument('--lat-min', default=0, type=int, help='minimum latitude index')
    parser.add_argument('--lat-max', default=45, type=int, help='maximum latitude index')
    parser.add_argument('--lon-min', default=0, type=int, help='minimum longtitude index')
    parser.add_argument('--lon-max', default=21, type=int, help='maximum longtitude index')

    # TRAINING
    parser.add_argument('--normalize-y', action='store_true', help='set target value between 0 and 1')
    parser.add_argument('--seq-length', default=6, type=int, help='length of training sequence (default 6)')
    # parser.add_argument('--eval-timesteps', default=12, type=int, help='evaluate on these timesteps')
    parser.add_argument('--num-epochs', default=50*2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num-samples', default=10, type=int, help='number of points in each timestep')
    parser.add_argument('--future', default=0, type=int, help='train on 1 + future timesteps')
    parser.add_argument('--epoch', default=-2, type=int, help='load from this epoch.pt file (-2 means gen data and scratch train, -1 means only scratch train')

    # LOGGING
    parser.add_argument('--save-every', default=1, type=int, help='save results every ... epochs')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--logid', default=None, type=str, help='unique id for each experiment')
    parser.add_argument('--runid', default=0, type=int, help='unique run id for each experiment')
    
    args = parser.parse_args()
    # setup logdir
    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    args.logdir = os.path.join(args.logdir, str(logid))
    args.runid = str(args.runid)

    return args

