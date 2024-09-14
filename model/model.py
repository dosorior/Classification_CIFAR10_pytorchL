import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import lightning as L
from torch.nn import functional as F
from torch import nn
from torchvision.models import resnet34


class LitResnet(L.LightningModule):
    def __init__(self,transfer=False):
        super().__init__()
        self.loss_criteria = nn.CrossEntropyLoss()
        self.model = resnet34(pretrained=transfer)

        if transfer:
             # layers are frozen by using eval()
            self.model.eval()
            # freeze params
            for param in self.model.parameters():
                param.requires_grad = False        

        else:

            self.num_ftrs = self.model.fc.in_features
            # print(f"The number of features of the fc is { self.num_ftrs}")
            self.model.fc = nn.Linear(self.num_ftrs,10) #change the output of classes
  
    
    def forward(self, imgs):
        return self.model(imgs)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        train,labels = train_batch
        preds = self.model(train)
        crit_loss = self.loss_criteria(preds,labels) 
        self.log("train_loss", crit_loss, prog_bar=True, on_step=False, on_epoch=True)
        return crit_loss
    
    def validation_step(self, val_batch, batch_idx):
        train,labels = val_batch
        preds = self.model(train)
        crit_loss = self.loss_criteria(preds,labels) 
        self.log('val_loss', crit_loss, prog_bar=True, on_step=True, on_epoch=True) 
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_acc", acc,  prog_bar=True, on_step=False, on_epoch=True)
        return crit_loss

    def test_step(self, val_batch, batch_idx):
        x,y = val_batch
        x_hat = self.model(x)
        crit_loss = self.loss_criteria(x_hat,y)  ## you can define the number of ephocs
        self.log('test_loss', crit_loss, prog_bar=True)
        acc = (y == x_hat).float().mean()
        self.log("test_acc", acc,  prog_bar=True, on_step=False, on_epoch=True)
    
    def create_model(pretrained):
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(512,10)
        return model



