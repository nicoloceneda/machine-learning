# %% SETUP
 
# Import the libraries

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Use GPU if available

print('='*20)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
print('='*20)


# %% DATA

# Prepare the dataset

class MnistDataModule(pl.LightningDataModule):     

    def __init__(self, data_path='datasets'):

        super().__init__()

        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        
        MNIST(root=self.data_path, download=True)

    def setup(self, stage=None): # stage: 'fit', 'validate', 'test', 'predict'

        mnist_all = MNIST(root=self.data_path, train=True, transform=self.transform, download=False)
        self.train, self.val = random_split(mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1))
        self.test = MNIST(root=self.data_path, train=False, transform=self.transform, download=False)

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=64, num_workers=0)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=64, num_workers=0)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=64, num_workers=0)

# Initialize the data module

torch.manual_seed(1)
mnist_dm = MnistDataModule()


# %% MODEL

# Design the multilayer perceptron

class MultiLayerPerceptron(pl.LightningModule):

    def __init__(self, image_shape=(1, 28, 28), hidden_size=(32, 16), output_size=10):

        super().__init__()

        self.train_acc = Accuracy(task='multiclass', num_classes=output_size)
        self.valid_acc = Accuracy(task='multiclass', num_classes=output_size)
        self.test_acc = Accuracy(task='multiclass', num_classes=output_size)

        input_size = image_shape[0] * image_shape[1] * image_shape[2]

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], output_size),
        )
    
    def forward(self, x):

        x = self.model(x)

        return x
    
    def training_step(self, batch, batch_idx):

        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):

        self.log('train_acc', self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        
        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):

        self.log('valid_acc', self.valid_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()
    
    def test_step(self, batch, batch_idx):

        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self):

        self.log('test_acc', self.test_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc.reset()
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer

# Initialize a multilayer perceptron object

model = MultiLayerPerceptron()


# %% TRAINING

# Learn from the data

tb_logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name='', version=0)

if torch.cuda.is_available():

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=tb_logger)

elif torch.backends.mps.is_available():

    trainer = pl.Trainer(max_epochs=10, accelerator='mps', devices=1, logger=tb_logger)

else:

    trainer = pl.Trainer(max_epochs=10, logger=tb_logger)

trainer.fit(model=model, datamodule=mnist_dm)

# Resume training from checkpoint and train for 5 additional epochs

if torch.cuda.is_available():

    trainer = pl.Trainer(max_epochs=15, accelerator='gpu', devices=1, logger=tb_logger)

elif torch.backends.mps.is_available():

    trainer = pl.Trainer(max_epochs=15, accelerator='mps', devices=1, logger=tb_logger)

else:

    trainer = pl.Trainer(max_epochs=15, logger=tb_logger)

ckpt_path = './lightning_logs/version_0/checkpoints/epoch=9-step=8600.ckpt'
trainer.fit(model=model, datamodule=mnist_dm, ckpt_path=ckpt_path)

# For model inspection run the terminal command: tensorboard --logdir lightning_logs/

# %% TESTING

# Evaluate the model on the test set

trainer.test(model=model, datamodule=mnist_dm)
