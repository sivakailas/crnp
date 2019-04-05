import os
import argparse
import datetime


def get_args():
    parser = argparse.ArgumentParser()

    # DATASET
    parser.add_argument('--mnist', action='store_true', help='use moving mnist dataset')
    parser.add_argument('--data-file', default='noaa_datasets/air.mon.mean.nc')
    parser.add_argument('--random-roi', action='store_true', help='A random region of interest is sampled at every instant')
    parser.add_argument('--roi-size', default=30, type=int, help='size of region of interest')
    # these are indices, check the data file to see the actual values of them
    parser.add_argument('--lat-min', default=15, type=int, help='minimum latitude index')
    parser.add_argument('--lat-max', default=45, type=int, help='maximum latitude index')
    parser.add_argument('--lon-min', default=95, type=int, help='minimum longtitude index')
    parser.add_argument('--lon-max', default=125, type=int, help='maximum longtitude index')

    # TRAINING
    parser.add_argument('--normalize-y', action='store_true', help='set target value between 0 and 1')
    parser.add_argument('--seq-length', default=6, type=int, help='length of training sequence (default 6)')
    # parser.add_argument('--eval-timesteps', default=12, type=int, help='evaluate on these timesteps')
    parser.add_argument('--num-epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-samples', default=100, type=int, help='number of points in each timestep')
    parser.add_argument('--future', default=0, type=int, help='train on 1 + future timesteps')

    # LOGGING
    parser.add_argument('--save-every', default=5, type=int, help='save results every ... epochs')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--logid', default=None, type=int, help='unique id for each experiment')
    
    args = parser.parse_args()
    # setup logdir
    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    args.logdir = os.path.join(args.logdir, str(logid))
    return args

