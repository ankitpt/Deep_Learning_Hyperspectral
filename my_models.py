# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from utilss import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)
    
    
    
    if name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs['patch_size']
        center_pixel = False
        filters=kwargs["filters"]
        model = LeeEtAl(n_bands, n_classes,filters=filters)
  #      lr = kwargs.setdefault('learning_rate', lr_choice)
        lr=kwargs['learning_rate']
        print("patch size, learning rate and filters in this model are ",patch_size,lr,filters)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    else:
        raise KeyError("{} model is unknown.".format(name))
        
        
class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes,filters):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
#        self.conv_3x3 = nn.Conv3d(
 #           1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#        self.conv_1x1 = nn.Conv3d(
#            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)
 
        self.conv_3x3 = nn.Conv3d(
                 1, filters, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, filters, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)


        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(filters*2, filters, (1, 1))
        self.conv2 = nn.Conv2d(filters, filters, (1, 1))
        self.conv3 = nn.Conv2d(filters, filters, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(filters, filters, (1, 1))
        self.conv5 = nn.Conv2d(filters, filters, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(filters, filters, (1, 1))
        self.conv7 = nn.Conv2d(filters, filters, (1, 1))
        self.conv8 = nn.Conv2d(filters, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(filters*2)
        self.lrn2 = nn.LocalResponseNorm(filters)
        self.bn1=nn.BatchNorm2d(filters*2)
        self.bn2=nn.BatchNorm2d(filters)
        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
     ##   print(x.size())
        x_3x3 = self.conv_3x3(x)
     #   print(x_3x3.size())
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
     #   x = F.relu(self.lrn1(x))
        x=F.relu(self.bn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
      #  x = F.relu(self.lrn2(x))
        x=F.relu(self.bn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        #print(x.size)
        return x
    
    
    
    
    
class LeeEtAl_without_residual(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes,filters):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
#        self.conv_3x3 = nn.Conv3d(
 #           1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#        self.conv_1x1 = nn.Conv3d(
#            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)
 
        self.conv_3x3 = nn.Conv3d(
                 1, filters, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, filters, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)


        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(filters*2, filters, (1, 1))
        self.conv2 = nn.Conv2d(filters, filters, (1, 1))
        self.conv3 = nn.Conv2d(filters, filters, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(filters, filters, (1, 1))
        self.conv5 = nn.Conv2d(filters, filters, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(filters, filters, (1, 1))
        self.conv7 = nn.Conv2d(filters, filters, (1, 1))
        self.conv8 = nn.Conv2d(filters, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(filters*2)
        self.lrn2 = nn.LocalResponseNorm(filters)
        self.bn1=nn.BatchNorm2d(filters*2)
        self.bn2=nn.BatchNorm2d(filters)
        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
     ##   print(x.size())
        x_3x3 = self.conv_3x3(x)
     #   print(x_3x3.size())
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
     #   x = F.relu(self.lrn1(x))
        x=F.relu(self.bn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
      #  x = F.relu(self.lrn2(x))
        x=F.relu(self.bn2(x))

        # First residual block
        #x_res = F.relu(self.conv2(x))
        #x_res = self.conv3(x_res)
        #x = F.relu(x + x_res)

        # Second residual block
        #x_res = F.relu(self.conv4(x))
        #x_res = self.conv5(x_res)
        #x = F.relu(x + x_res)
        
        # First residual block
        x_res = self.conv2(x)
        x_res = self.conv3(x_res)
        x = F.relu(x_res)

        # Second residual block
        x_res = self.conv4(x)
        x_res = self.conv5(x_res)
        x = F.relu(x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        #print(x.size)
        return x    
    
    
    
############Adding modified leeetal

class LeeEtAl_modified(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl_modified, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_3x3_3x3 = nn.Conv3d(
            128, 128, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        #self.conv0 = nn.conv2d(128 , 128,(1,1))
        self.conv1 = nn.Conv2d(384, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
       # print(x.size())
        x_3x3_layer_1 = self.conv_3x3(x)
       # print(x_3x3_layer_1.size())
        x_3x3_3x3 = self.conv_3x3_3x3(x_3x3_layer_1)
       # print(x_3x3_3x3.size())
        
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3_3x3, x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)
       
        # Inception module
        #x_3x3 = self.conv_3x3(x)
        #x_1x1 = self.conv_1x1(x)
        #x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        # x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x        




def train(dataset_name,net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=1, device=torch.device('cpu'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
        dataset_name: Name of the dataset
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    val_max=-10000000 
    loss_epochs=[]
    not_improved=0
    patience=10
 #   for e in tqdm(range(1, epoch + 1), desc="Training the network"):
    for e in range(1, epoch + 1):
    
        # Set the network to training mode
        #net = net.float()
        net.train()
        avg_loss = 0.
        
        # Run the training loop for one epoch
       # for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for batch_idx, (data, target) in enumerate(data_loader):

            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == 'full':
                
                output = net(data)
               # print("output size is ", output.dtype)
                #print("target size is ",target.dtype)
                loss = criterion(output, target)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            
            #Backprop to find gradients
            loss.backward()
            
            #Update paramters via gradients from bp
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            

            if len(val_accuracies) > 0 and display is not None:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
           # print(iter_)
            iter_ += 1
            del(data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        loss_epochs.append(avg_loss)
        print("%d %d %f"%(e,iter_+1,avg_loss))
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc

            if(val_acc>val_max):
                
                val_max=val_acc
               # save_model(net, camel_to_snake(str(net.__class__.__name__)), dataset_name, epoch=e, metric=abs(metric))
                best_model=net
                save_model(net, camel_to_snake(str(net.__class__.__name__)), dataset_name, epoch=e, metric=abs(metric))
                not_improved=0
            else:
                
                not_improved=not_improved+1
                if(not_improved>patience):
                    
                    print("Max patience reached, stopping training")
                    print("Best validation accuracy reached in this model is %.3f"%val_max)
                    break
            
            
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
       # if e % save_epoch == 0:
            
  
    return best_model,val_max,val_accuracies,loss_epochs        


def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
     if isinstance(model, torch.nn.Module):
         filename = str('wk') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
     else:
         filename = str('wk')
         tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

def val(net, data_loader, device='cpu', supervision='full'):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            #print(output.size())
            for out, pred in zip(output.view(-1), target.view(-1)):
                if pred.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


