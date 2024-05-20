import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from mne.viz import circular_layout
from torcheeg.models import CCNN
import torch.nn as nn
from torcheeg.models import DGCNN
from pylab import cm
import torch
import os
from torcheeg.models.pyg import RGNN
from torcheeg.trainers import ClassificationTrainer
from torch_geometric.loader import DataLoader
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
import numpy as np
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
from torcheeg.transforms.pyg import ToG
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
train_type = 'DGCNN_train_ica__final'
file_path = os.path.join("./",train_type,"model.pth")
epochs = 50


class MyClassificationTrainer(ClassificationTrainer):
    def __init__(self, model: nn.Module, trainer_k:str=None, num_classes=None, lr: float = 0.0, weight_decay: float = 0,optimizer = None, **kwargs):
        super().__init__(model, num_classes, lr, weight_decay, **kwargs)
        self.writer = SummaryWriter(f"./{train_type}/train_{trainer_k}/loss")
        self.steps_file_name =f"./{train_type}/train_{trainer_k}/steps"
        try:
            model.load_state_dict(torch.load(os.path.join(train_type,f'train_{trainer_k}','model.pth')))
        except FileNotFoundError:
            pass
        self.train_counter = 0
        self.test_counter = 0
        self.trainer_k = trainer_k
        self.optimizer = optimizer
        self.epoch = 0
        self.last_train_loss = 0
        self.last_train_accuracy = 0
        self.cmdata_file = open(f"./{train_type}/train_{trainer_k}/metrics_{trainer_k}","a+")

    def on_training_step(self, train_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        super().on_training_step(train_batch, batch_id, num_batches, **kwargs)
        if self.train_loss.mean_value.item() != 0:
            self.last_train_loss = self.train_loss.compute()
            self.last_train_accuracy = self.train_accuracy.compute()

    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        super().after_validation_epoch(epoch_id, num_epochs, **kwargs)
        torch.save(model.state_dict(),f"./{train_type}/train_{self.trainer_k}/model.pth")
        self.writer.add_scalars('loss', {
            'train': self.last_train_loss,
            'validation': self.val_loss.compute()
        }, self.train_counter)
        self.writer.add_scalars('accuracy', {
            'train': self.last_train_accuracy*100,
            'validation': self.val_accuracy.compute()*100
        }, self.train_counter)
        self.train_counter += 1
        self.epoch += 1

    def after_test_epoch(self, **kwargs):
        super().after_test_epoch(**kwargs)
        self.writer.add_scalar("loss/test", self.test_loss.compute(), self.test_counter)
        self.writer.add_scalar("accuracy/test", self.test_accuracy.compute()*100, self.test_counter)
        self.test_counter += 1

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        pred = self.modules['model'](X)

        data = np.column_stack((pred.tolist(), y))
        for row in data:
            line = ", ".join([f"{elem}" for elem in row])
            self.cmdata_file.write(f"{line}\n")

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)


def applying_ICA(tensore):
    ica = FastICA(n_components=62, random_state=10, max_iter=4000, tol=1e-4)
    ica_result = ica.fit_transform(tensore)
    return ica_result


def find_weight(x):
    abs_components = np.abs(x)
    top5_indices = np.argsort(-abs_components, axis=1)[:, :5]
    return x

# Usage of the new dataset class
if __name__ == '__main__':
    dataset = SEEDIVDataset(io_path='./dataset/seed_iv_ICA_2',
                            root_path='./dataset/eeg_raw_data',
                            online_transform=transforms.Compose([
                                transforms.BandDifferentialEntropy(
                                    band_dict={
                                            "delta": [1, 4],
                                            "theta": [4, 8],
                                            "alpha": [8, 14],
                                            "beta": [14, 31],
                                            "gamma": [31, 49]
                                    }),
                                transforms.ToTensor()
                            ]),
                            label_transform=transforms.Select('emotion'),
                            chunk_size=800, 
                            num_worker=8)
    
    k_fold = KFoldGroupbyTrial(n_splits=5,shuffle=True,random_state=10,split_path='./dataset/splits_ica_dgcnn_final')
    # Addestramento della rete neurale
    for i, (train_dataset, val_dataset) in track(enumerate(k_fold.split(dataset)), "[bold green]Training: ", total=5):
        model = DGCNN(num_electrodes=62, in_channels=5, num_layers=2, hid_channels=32, num_classes=4).to(device)
        trainer = MyClassificationTrainer(model=model, trainer_k = i,optimizer = torch.optim.Adam(model.parameters(),lr = 0.001, weight_decay=0.001))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        trainer.fit(train_loader, val_loader, num_epochs=epochs)    
        trainer.test(val_loader)


    print('[bold green]Addestramento completato!')

    