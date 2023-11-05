import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, fs, order=6):
    """
    Applies a lowpass Butterworth filter to the data.

    Parameters:
    - data (array-like): The input signal to be filtered.
    - cutoff (float): The cutoff frequency of the filter.
    - fs (float): The sampling frequency of the data.
    - order (int, optional): The order of the filter. Default is 6.

    Returns:
    - y (array-like): The filtered signal.
    """

    # Design the Butterworth filter
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')

    # Apply the filter to the data using filtfilt (to avoid phase shift)
    y = filtfilt(b, a, data)

    return y


def get_test_dataloader(p, dataset):
    train_sampler = DistributedSampler(dataset) if p['use_ddp'] else None
    return torch.utils.data.DataLoader(
                            dataset,
                            num_workers=p["num_workers"],
                            batch_size=p["batch_size"],
                            shuffle=True,
                            drop_last=False,
                            sampler=train_sampler
                            ), train_sampler


def get_customized_dataset(data, label, mixup=False):
    if mixup:
        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, data, label):
                self.data = data
                self.label = label

            def __getitem__(self, index):
                x1 = self.data[index]
                y1 = self.label[index]
                index2 = np.random.randint(0, len(self.data))
                while index == index2:
                    index2 = np.random.randint(0, len(self.data))
                x2 = self.data[index2]
                y2 = self.label[index2]
                lam = np.random.beta(1, 1)
                x = lam * x1 + (1 - lam) * x2
                return x, (y1, y2, lam)

            def __len__(self):
                return len(self.data)

        return MyDataset(torch.from_numpy(np.abs(data)).float(), torch.from_numpy(label).long())

    else:
        return torch.utils.data.TensorDataset(torch.from_numpy(np.abs(data)).float(), torch.from_numpy(label).long())


def load_data_and_get_dataloader(data_path, config, mixup=False):
    file = np.load(data_path)
    data, label = file['x'], file['y']
    dataset = get_customized_dataset(data, label, mixup=mixup)
    dataloader, sampler = get_test_dataloader(config, dataset)
    return dataloader, sampler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_counter = 0

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.save_counter += 1
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + 'pretrained_{}.pth'.format(self.save_counter))
        with open(self.path + 'pretrained_logs.txt', 'a+') as f:
            f.write('Current save time: ' + str(self.save_counter) + ', Loss: ' + str(val_loss) + '\n')
        self.val_loss_min = val_loss
