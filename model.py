import torch
import torch.nn as nn
import ipdb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # nn.init.orthogonal_(m.weight.data)
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(self, layer_sizes, output_size):
        super().__init__()

        self.actv = nn.LeakyReLU()

        modules = []
        for i in range(len(layer_sizes)-1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            modules.append(self.actv)
        modules.append(nn.Linear(layer_sizes[-1], output_size))

        self.layers = nn.Sequential(*modules)

    def forward(self, inp):
        return self.layers(inp)


class Forecaster(nn.Module):
    def __init__(self, input_size=128):
        super(Forecaster, self).__init__()
        self.num_lstm_layers = 1
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.lstm_hidden_dim, num_layers=self.num_lstm_layers)
        self.linear = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(device), 
                torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(device)) 

    def forward(self, inp, future=0):
        batch_size = inp.size(1)
        hidden = self.init_hidden(batch_size, inp.device)
        for i in inp:
            lstm_out, hidden = self.lstm(i.view(1, batch_size, -1), hidden)
            # out = self.linear(lstm_out)
            # outputs.append(out)

        out = self.linear(lstm_out)
        outputs = [out]
        # for future prediction, feed the prediction as input to the next timestep
        for i in range(future):
            lstm_out, hidden = self.lstm(out, hidden)
            out = self.linear(lstm_out)
            outputs.append(out)
        outputs = torch.cat(outputs)
        return outputs


class CRNP(nn.Module):
    def __init__(self, input_size, output_size, sigmoid_output=False):
        super().__init__()
        embed_size = 128
        hidden_size = 128
        self.output_size = output_size
        enc_sizes = [input_size + output_size] + [hidden_size]*3
        dec_sizes = [embed_size + input_size] + [hidden_size]*3 
        self.encoder = MLP(enc_sizes, embed_size)
        self.forecaster = Forecaster(embed_size)
        self.decoder = MLP(dec_sizes, 2*output_size)
        self.sigmoid_output = sigmoid_output
        self.apply(weights_init)

    def _compute_loss(self, r_next, target_x, target_y=None):
        r_next = r_next.permute([1,0,2]).repeat(1,target_x.shape[1],1)
        x = torch.cat([r_next, target_x], dim=-1)

        out = self.decoder(x)
        mu, logsigma = torch.split(out, self.output_size, dim=-1)
        mu = mu.squeeze(-1)
        logsigma = logsigma.squeeze(-1)
        if self.sigmoid_output:
            mu = torch.sigmoid(mu)
        sigma = 0.1 + 0.9 * torch.nn.Softplus()(logsigma)
        dist = torch.distributions.Normal(mu, sigma)

        return mu, sigma, dist

    def forward(self, context_x, context_y, target_x, target_y=None, target_inp=None):
        # (batch, timesteps, num_points, dim)
        context_pairs = torch.cat([context_x, context_y], -1)
        
        # TODO: this encoder can be a multi-head attention module (aka, graph network)
        x = self.encoder(context_pairs)
        
        # (timesteps, batch, dim)
        r = torch.mean(x, dim=-2).permute([1,0,2]) 
        r_next = self.forecaster(r)
        
        # loopwise implementation
        # mu, sigma, loss = self._compute_loss(r_next, target_x, target_y)
        # if target_inp is not None:
        #     # all_mu = []
        #     # all_sigma = []
        #     for i in range(len(r)):
        #         inp_mu, inp_sigma, inp_loss = self._compute_loss(r[i:i+1], target_x, target_inp[:,i:i+1,:])
        #         # all_mu.append(inp_mu.detach().cpu())
        #         # all_sigma.append(inp_sigma.detach().cpu())
        #         loss += inp_loss

        # predict on the inp steps also
        # batchwise implementation
        if target_inp is not None:
            # this is bottleneck for 
            r_all = torch.cat([r, r_next], dim=0)
            # (batch_size, num_timesteps, num_points, dim)
            r_all = r_all.permute([1,0,2]).unsqueeze(2).repeat(1,1,target_x.shape[-2],1)
            
            # (batch_size, num_timesteps, num_points, 2)
            all_target_x = target_x.unsqueeze(1).repeat(1,r_all.shape[1],1,1)
            x = torch.cat([r_all, all_target_x], dim=-1)

        else:
            # (batch, num_target, dim)
            r_next = r_next.permute([1,0,2]).repeat(1,target_x.shape[1],1)
            x = torch.cat([r_next, target_x], dim=-1)

        out = self.decoder(x)
        mu, logsigma = torch.split(out, self.output_size, dim=-1)
        mu = mu.squeeze(-1)
        logsigma = logsigma.squeeze(-1)
        if self.sigmoid_output:
            mu = torch.sigmoid(mu)
        sigma = 0.1 + 0.9 * torch.nn.Softplus()(logsigma)
        dist = torch.distributions.Normal(mu, sigma)

        return mu, sigma, dist