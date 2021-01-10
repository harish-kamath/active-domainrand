from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)
def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)
def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)
def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)
def identity(x):
    return x

class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b

class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            layer_norm=False,
            layer_norm_kwargs=None,
            spectral_norm=False,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # data normalization
        self.input_mu = nn.Parameter(
            zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            ones(input_size), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            layer_size = (ensemble_size, in_size, next_size)
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                w_std_value=1/(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            if spectral_norm:
                fc = nn.utils.spectral_norm(fc, name='W')
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            w_std_value=1/(2*np.sqrt(in_size)),
            b_init_value=b_init_value,
        )

    def forward(self, input):
        dim = len(input.shape)

        # input normalization
        h = (input - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, output_size)
        return output

    def sample(self, input):
        preds = self.forward(input)

        inds = torch.randint(0, len(self.elites), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=device)
        inds = inds.repeat(1, preds.shape[2])

        samples = (inds == 0).float() * preds[self.elites[0]]
        for i in range(1, len(self.elites)):
            samples += (inds == i).float() * preds[self.elites[i]]

        return samples

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = from_numpy(mean)
        self.input_std.data = from_numpy(std)

class ProbabilisticEnsemble(ParallelizedEnsemble):

    """
    Probabilistic ensemble (Chua et al. 2018).
    Implementation is parallelized such that every model uses one forward call.
    Each member predicts the mean and variance of the next state.
    Sampling is done either uniformly or via trajectory sampling.
    """

    def __init__(
            self,
            ensemble_size,        # Number of members in ensemble
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_sizes,         # Hidden sizes for each model
            spectral_norm=False,  # Apply spectral norm to every hidden layer
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=2*(obs_dim + 2),  # We predict (reward, done, next_state - state)
            hidden_activation=torch.tanh,
            spectral_norm=spectral_norm,
            **kwargs
        )

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.output_size = obs_dim + 2

        # Note: we do not learn the logstd here, but some implementations do
        self.max_logstd = nn.Parameter(
            ones(obs_dim + 2), requires_grad=False)
        self.min_logstd = nn.Parameter(
            -ones(obs_dim + 2) * 5, requires_grad=False)

    def forward(self, input, deterministic=False, return_dist=True):
        output = super().forward(input)
        mean, logstd = torch.chunk(output, 2, dim=-1)

        # Variance clamping to prevent poor numerical predictions
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        if deterministic:
            return mean, logstd if return_dist else mean

        std = torch.exp(logstd)
        eps = randn(std.shape)
        samples = mean + std * eps

        if return_dist:
            return samples, mean, logstd
        else:
            return samples

    def get_loss(self, x, y, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state

        mean, logstd = self.forward(x, deterministic=True, return_dist=True)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        # Maximize log-probability of transitions
        inv_var = torch.exp(-2 * logstd)
        sq_l2_error = (mean - y)**2
        if return_l2_error:
            l2_error = torch.sqrt(sq_l2_error).mean(dim=-1).mean(dim=-1)

        loss = (sq_l2_error * inv_var + 2 * logstd).sum(dim=-1).mean(dim=-1)

        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:
            if return_l2_error:
                return loss.mean(), l2_error.mean()
            else:
                return loss.mean()

    def sample_with_disagreement(self, input, return_dist=False, disagreement_type='mean'):
        preds, mean, logstd = self.forward(input, deterministic=False, return_dist=True)

        # Standard uniformly from the ensemble
        inds = torch.randint(0, preds.shape[0], input.shape[:-1])

        # Ensure we don't use the same member to estimate disagreement
        inds_b = torch.randint(0, mean.shape[0], input.shape[:-1])
        inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0])

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=device)
        inds = inds.repeat(1, preds.shape[2])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=device)
        inds_b = inds_b.repeat(1, preds.shape[2])

        # Uniformly sample from ensemble
        samples = (inds == 0).float() * preds[0]
        for i in range(1, preds.shape[0]):
            samples += (inds == i).float() * preds[i]

        if disagreement_type == 'mean':
            # Disagreement = mean squared difference in mean predictions (Kidambi et al. 2020)
            means_a = (inds == 0).float() * mean[0]
            means_b = (inds_b == 0).float() * mean[0]
            for i in range(1, preds.shape[0]):
                means_a += (inds == i).float() * mean[i]
                means_b += (inds_b == i).float() * mean[i]

            disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

        elif disagreement_type == 'var':
            # Disagreement = max Frobenius norm of covariance matrix (Yu et al. 2020)
            vars = (2 * logstd).exp()
            frobenius = torch.sqrt(vars.sum(dim=-1))
            disagreements, *_ = frobenius.max(dim=0)
            disagreements = disagreements.reshape(-1, 1)

        else:
            raise NotImplementedError

        if return_dist:
            return samples, disagreements, mean, logstd
        else:
            return samples, disagreements

class MBRLTrainer:
    def __init__(
            self,
            ensemble,
            num_elites=None,
            learning_rate=1e-3,
            batch_size=256,
            optimizer_class=optim.Adam,
            train_call_freq=1,
            **kwargs
    ):
        super().__init__()

        self.ensemble = ensemble
        self.ensemble_size = ensemble.ensemble_size
        self.num_elites = min(num_elites, self.ensemble_size) if num_elites \
                          else self.ensemble_size

        self.obs_dim = ensemble.obs_dim
        self.action_dim = ensemble.action_dim
        self.batch_size = batch_size
        self.train_call_freq = train_call_freq

        self.optimizer = self.construct_optimizer(
            ensemble, optimizer_class, learning_rate)

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def construct_optimizer(self, model, optimizer_class, lr):
        decays = [.000025, .00005, .000075, .000075, .0001]

        fcs = model.fcs + [model.last_fc]
        optimizer = optimizer_class([
                {'params': fcs[i].parameters(), 'weight_decay': decays[i]} \
                    for i in range(len(fcs))
            ], lr=lr
        )

        return optimizer

    def train_from_buffer(self, data, holdout_pct=0.2, max_grad_steps=1000, epochs_since_last_update=5):
        self._n_train_steps_total += 1
        if self._n_train_steps_total % self.train_call_freq > 0 and self._n_train_steps_total > 1:
            return
        
        x = data[:,:self.obs_dim + self.action_dim]  # inputs  s, a
        y = data[:,self.obs_dim + self.action_dim:]  # predict r, d, ns
        y[:,-self.obs_dim:] -= x[:,:self.obs_dim]    # predict delta in the state

        # normalize network inputs
        self.ensemble.fit_input_stats(x)

        # generate holdout set
        inds = np.random.permutation(data.shape[0])
        x, y = x[inds], y[inds]

        n_train = max(int((1-holdout_pct) * data.shape[0]), data.shape[0] - 8092)
        n_test = data.shape[0] - n_train

        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]
        x_test, y_test = from_numpy(x_test), from_numpy(y_test)

        # train until holdout set convergence
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        best_holdout_loss = float('inf')
        num_batches = int(np.ceil(n_train / self.batch_size))

        while num_epochs_since_last_update < epochs_since_last_update and num_steps < max_grad_steps:
            # generate idx for each model to bootstrap
            self.ensemble.train()
            for b in range(num_batches):
                b_idxs = np.random.randint(n_train, size=(self.ensemble_size*self.batch_size))
                x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]
                x_batch, y_batch = from_numpy(x_batch), from_numpy(y_batch)
                x_batch = x_batch.view(self.ensemble_size, self.batch_size, -1)
                y_batch = y_batch.view(self.ensemble_size, self.batch_size, -1)
                loss = self.ensemble.get_loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                wandb.log({"Dynamics Loss": loss.detach()})
                self.optimizer.step()
            num_steps += num_batches

            # stop training based on holdout loss improvement
            self.ensemble.eval()
            with torch.no_grad():
                holdout_losses, holdout_errors = self.ensemble.get_loss(
                    x_test, y_test, split_by_model=True, return_l2_error=True)
            holdout_loss = sum(sorted(holdout_losses)[:self.num_elites]) / self.num_elites
            wandb.log({"Dynamics Holdout Loss": np.mean(get_numpy(sum(holdout_losses))) / self.ensemble_size})

            if num_epochs == 0 or \
               (best_holdout_loss - holdout_loss) / abs(best_holdout_loss) > 0.01:
                best_holdout_loss = holdout_loss
                num_epochs_since_last_update = 0
            else:
                num_epochs_since_last_update += 1

            num_epochs += 1

        self.ensemble.elites = np.argsort(holdout_losses)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['Model Elites Holdout Loss'] = \
                np.mean(get_numpy(holdout_loss))
            self.eval_statistics['Model Holdout Loss'] = \
                np.mean(get_numpy(sum(holdout_losses))) / self.ensemble_size
            self.eval_statistics['Model Training Epochs'] = num_epochs
            self.eval_statistics['Model Training Steps'] = num_steps

            for i in range(self.ensemble_size):
                name = 'M%d' % (i+1)
                self.eval_statistics[name + ' Loss'] = \
                    np.mean(get_numpy(holdout_losses[i]))
                self.eval_statistics[name + ' L2 Error'] = \
                    np.mean(get_numpy(holdout_errors[i]))

    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.ensemble
        ]

    def get_snapshot(self):
        return dict(
            ensemble=self.ensemble
        )