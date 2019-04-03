import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h)
    cols = torch.linspace(0, 1, w)
    x, y = torch.meshgrid(rows, cols)
    grid = torch.stack([x.flatten(),y.flatten()]).t()
    # grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def load_nc_data(data_file, variable='air'):
    from netCDF4 import Dataset as dt
    f = dt(data_file)
    if variable=='air':
        air = f.variables['air']
        air_range = air.valid_range
        data = air[:].data
        # convert to degree celsius
        if air.units == 'degK':
            data -= 273
            air_range -= 273
    else:
        precip = f.variables['precip']
        data = precip[:].data
    return data


def save_image(inps, true, mu, fn, var=None):
    # TODO: find a way to determine figsize
    # compute this from the size of image
    fig, ax = plt.subplots(2, 4, figsize=(16,8), sharex=True, sharey=True)
    ax = ax.flatten()
    for ax_ in ax:
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)
    
    n = len(inps)
    vmin = true.min()
    vmax = true.max()
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    sns.heatmap(inps[0], ax=ax[0], cmap='ocean', vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax, square=False)
    ax[0].set_title('T=1')
    for i in range(1, n):
        sns.heatmap(inps[i], ax=ax[i], cmap='ocean', vmin=vmin, vmax=vmax, cbar=False, square=False)
        ax[i].set_title('T={}'.format(i+1))

    sns.heatmap(true, ax=ax[n], cmap='ocean', vmin=vmin, vmax=vmax, cbar=False, square=False)
    ax[n].set_title('Actual (T={})'.format(n+1))

    sns.heatmap(mu, ax=ax[n+1], cmap='ocean', vmin=vmin, vmax=vmax, cbar=False, square=False)
    ax[n+1].set_title('Prediction (T={})'.format(n+1))

    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.savefig(fn)
    plt.close(fig)
