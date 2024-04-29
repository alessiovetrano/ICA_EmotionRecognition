import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from mne.viz import circular_layout
from torcheeg.models import CCNN
import torch.nn as nn
from pylab import cm
import torch
import os
from torcheeg.trainers import ClassificationTrainer
from torch_geometric.loader import DataLoader
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
import numpy as np
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from torcheeg.utils import plot_raw_topomap, plot_2d_tensor, plot_signal
from torchvision import datasets as T
import matplotlib.pyplot as plt
from torcheeg import transforms
from torcheeg.datasets import SEEDIVDataset
from rich.progress import track
from rich import print
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt
import warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
from typing import List, Tuple
from torch.utils.tensorboard.writer import SummaryWriter
from torcheeg.datasets.constants.emotion_recognition.seed_iv import SEED_IV_STANDARD_ADJACENCY_MATRIX, SEED_IV_CHANNEL_LOCATION_DICT


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
epochs = 20
print(f'DEVICE USED: {device}')
train_type = 'result_train_CNN_ICA_30E_cm'
file_path = os.path.join("./",train_type,"model.pth")

class MyClassificationTrainer(ClassificationTrainer):
    def __init__(self, index, model: nn.Module, num_classes: int | None = None, lr: float = 0.0001, weight_decay: float = 0, device_ids: List[int] = [], ddp_sync_bn: bool = True, ddp_replace_sampler: bool = True, ddp_val: bool = True, ddp_test: bool = True):
        super().__init__(model, num_classes, lr, weight_decay, device_ids, ddp_sync_bn, ddp_replace_sampler, ddp_val, ddp_test)
        self.writer = SummaryWriter(f"train_final/train_{index}/metrics")
        try:
            model.load_state_dict(torch.load(f"train_final/train_{index}/model.pth"))
        except FileNotFoundError:
            pass
        self.index = index
        self.epoch = 0
        self.last_train_loss = 0
        self.last_train_accuracy = 0
        self.cmdata_file = open(f"train_final/train_{index}/cmdata", "a+")

    def on_training_step(self, train_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        super().on_training_step(train_batch, batch_id, num_batches, **kwargs)
        if self.train_loss.mean_value.item() != 0:
            self.last_train_loss = self.train_loss.compute()
            self.last_train_accuracy = self.train_accuracy.compute()
    
    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        super().after_validation_epoch(epoch_id, num_epochs, **kwargs)
        torch.save(model.state_dict(), f"train_final/train_{self.index}/model.pth")
        self.writer.add_scalars('loss', {
            'train': self.last_train_loss,
            'validation': self.val_loss.compute()
        }, self.epoch)
        self.writer.add_scalars('accuracy', {
            'train': self.last_train_accuracy*100,
            'validation': self.val_accuracy.compute()*100
        }, self.epoch)
        self.epoch += 1
    
    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        pred = self.modules['model'](X)
        data = np.column_stack((pred.tolist(), y))
        for row in data:
            line = ", ".join([f"{elem}" for elem in row])
            self.cmdata_file.write(f"{line}\n")

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)

    def after_test_epoch(self, **kwargs):
        super().after_test_epoch(**kwargs)
        self.writer.add_scalar("loss/test", self.test_loss.compute(), self.epoch)
        self.writer.add_scalar("accuracy/test", self.test_accuracy.compute()*100, self.epoch)

def applying_ICA(tensore):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        ica = FastICA(n_components=62, random_state=10, max_iter=4000, tol=1e-4)
        ica_result = ica.fit_transform(tensore)
    return ica_result




if __name__ == '__main__':
    dataset = SEEDIVDataset(io_path='./dataset/seed_iv_ICA_2',
                            root_path='./dataset/eeg_raw_data',
                            offline_transform=transforms.Lambda(lambda x:applying_ICA(x)),
                            online_transform=transforms.Compose([
                                transforms.BandDifferentialEntropy(band_dict={
                                            "delta": [1, 4],
                                            "theta": [4, 8],
                                            "alpha": [8, 14],
                                            "beta": [14, 31],
                                            "gamma": [31, 49]
                                        }),
                                transforms.MeanStdNormalize(),
                                transforms.ToGrid(SEED_IV_CHANNEL_LOCATION_DICT),
                                transforms.ToTensor()
                            ]),
                            label_transform=transforms.Select('emotion'))

    k_fold = KFoldGroupbyTrial(n_splits=5,shuffle=True,random_state=10,split_path='./dataset/splits_ica_30_final')


    for i, (train_dataset, val_dataset) in track(enumerate(k_fold.split(dataset)), "[bold green]Training: ", total=5):
        model = CCNN (in_channels=5, dropout=0.5, num_classes=4).to(device)
        trainer = MyClassificationTrainer(index=i, model=model, lr=0.001, weight_decay=0.0001)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        trainer.fit(train_loader, val_loader, num_epochs=epochs)    
        trainer.test(val_loader)
