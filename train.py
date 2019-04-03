from model import CRNP
import torch
from tensorboard_logger import log_value
import numpy as np
import utils as ut
import os
import ipdb


class Trainer:
    def __init__(self, train_dataloader, test_dataloader, args, data_mean=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_size = 2
        self.output_size = 1
        # if using mnist, use sigmoid in the decoding step
        self.net = CRNP(input_size=self.input_size,
                        output_size=self.output_size,
                        sigmoid_output=args.mnist).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)
        self.args = args
        self.sz1, self.sz2 = train_dataloader.dataset.data.shape[-2:]
        self.N = self.sz1 * self.sz2
        self.x_grid = ut.generate_grid(self.sz1, self.sz2)
        
        # used only for plotting partial images
        self.min_val = self.train_dataloader.dataset.data.min()
        self.data_mean = data_mean
        self.mean_reduction = data_mean is not None

    def select(self, inp, return_idx=False):
        # inp - num_batches x self.args.seq_length x 1 x self.sz1 x self.sz2

        # sample self.args.num_samples in each timestep
        sz = inp.shape
        inp_flat = inp.view(*sz[:-2], -1)
        idxs = torch.tensor(np.random.randint(0, self.N, (*sz[:-2], self.args.num_samples)))
        ys = torch.gather(inp_flat, -1, idxs).permute([0,1,3,2])
        x_all = self.x_grid.unsqueeze(1).expand(sz[0], sz[1], -1, -1)
        xs = x_all.gather(-2, idxs.permute([0,1,3,2]).expand(-1,-1,-1,self.input_size))
        if return_idx:
            return xs.to(self.device), ys.to(self.device), idxs
        return xs.to(self.device), ys.to(self.device)

    def save_model(self, epoch):
        checkpoint = {'model_state_dict': self.net.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'epoch': epoch,
                      }
        torch.save(checkpoint, os.path.join(self.args.logdir, str(epoch)+'.pt'))

    def train(self, num_epochs=50):
        step = 0
        for epoch in range(num_epochs):
            for batch_idx, (inp, target) in enumerate(self.train_dataloader):
                inp = inp.float()
                target = target.float()
                
                step += 1
                x_context, y_context = self.select(inp)
                x_target = self.x_grid.expand(len(inp),-1,-1).to(self.device)
                y_target = target.view(target.shape[0], -1, 1).to(self.device)

                self.optimizer.zero_grad()
                mu, sigma, dist, loss = self.net(x_context, y_context, x_target, y_target)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    mae = torch.mean(torch.abs(mu - y_target))

                print('Epoch {:d}/{:d} Batch {:d} Loss {:.3f} MAE {:.3f}'.format(
                      epoch, self.args.num_epochs, batch_idx, loss.item(), mae.item()))
                
                if not self.args.test:
                    log_value('train_mae', mae, step)
                    log_value('train_loss', loss.item(), step)

            save = (epoch+1)%self.args.save_every == 0
            self.test(epoch, save=save)

    def test(self, epoch, save=False):
        abs_err = 0
        count = 0
        if save:
            print('Saving model and results...')
            self.save_model(epoch)

        with torch.no_grad():
            for batch_idx, (inp, target) in enumerate(self.test_dataloader):
                x_context, y_context, idxs = self.select(inp, return_idx=True)
                x_target = self.x_grid.expand(len(inp),-1,-1).to(self.device)
                y_target = target.view(target.shape[0], -1, 1).to(self.device)
                
                mu, sigma, dist, loss = self.net(x_context, y_context, x_target, y_target)
                abs_err += torch.sum(torch.abs(mu - y_target))
                count += len(mu)
                
                if save:
                    t = np.random.randint(0, len(inp))
                    true = y_target[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                    pred = mu[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                    fn = os.path.join(self.args.logdir, 'epoch_' + str(epoch) + '_' + str(batch_idx) + '.png')
                    inps = self.reconstruct(y_context[t].cpu().numpy(), idxs[t].cpu().numpy())
                    if self.mean_reduction:
                        true += self.data_mean
                        pred += self.data_mean
                    ut.save_image(inps, true, pred, fn, var=None)

        test_mae = abs_err / (count * self.N)
        print('Test MAE: {:.6f}'.format(test_mae))
        log_value('test_mae', test_mae, epoch)

    def reconstruct(self, y_context, idxs):
        nb = len(y_context)
        canvas = [np.ones(self.N)*self.min_val for _ in range(nb)]
        for i in range(nb):
            idx = idxs[i].squeeze()
            if self.mean_reduction:
                canvas[i][idx] = y_context[i].squeeze() + self.data_mean.flatten()[idx]
            else:
                canvas[i][idx] = y_context[i].squeeze()
            canvas[i] = canvas[i].reshape(self.sz1, self.sz2) 
        return canvas