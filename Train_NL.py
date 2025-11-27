from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss
from torch_lib.Optimizer import NL_Optimizer

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torchvision.models import swin_v2_t, efficientnet_b0
from torch.utils import data
from torch_lib.data_aug import BatchOcclusion

import os
import argparse

def main():

    # hyper parameters
    epochs = 100
    batch_size = 4
    alpha = 0.6
    w = 0.4
    lr = 1e-4
    parser = argparse.ArgumentParser(description="Unified Training Script for Baseline (SGD) and Nested Learning")
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'nl'], 
                        help="Training mode: 'baseline' for SGD, 'nl' for Nested Learning")

    print("current mode:", parser.parse_args().mode)
    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    #randomly choose 50%    choose all dataset takes too long time
    total_size = len(dataset)
    keep_size = int(0.5 * total_size)
    keep_idx = np.random.choice(range(total_size), keep_size, replace=False)
    dataset = data.Subset(dataset, keep_idx)

    # split the subset into train/validation portions
    val_ratio = 0.2
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size],
                                                   generator=generator)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    train_gen = data.DataLoader(train_dataset, **params)
    val_params = dict(params)
    val_params['shuffle'] = False
    val_gen = data.DataLoader(val_dataset, **val_params)



    outer_model = vgg.vgg19_bn()
    #outer_model = efficientnet_b0()
    model = Model(features=outer_model.features).cuda()
    

    #main difference: choose optimizer based on mode
    args = parser.parse_args()
    if args.mode == 'baseline':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:  # Nested Learning
        terms = model.prepare_term_params(lr = lr)
        optimizer = NL_Optimizer(terms)
    data_aug = BatchOcclusion()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass


    #if latest_model is not None:
    #    checkpoint = torch.load(model_path + latest_model)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    first_epoch = checkpoint['epoch']
    #    loss = checkpoint['loss']
    #
    #    print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
    #    print('Resuming training....')



    total_num_batches = int(len(train_dataset) / batch_size)
    best_loss = float('inf')
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        loss_tot = 0
        for local_batch, local_labels in train_gen:

            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            
            #data augmentation
            local_batch = data_aug(local_batch)
            local_batch=local_batch.float().cuda()
            [orient, conf, dim, features, raw_orient] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            
            
            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta
            
            #if args.mode == 'nl': 
            #     loss += model.compute_nested_loss(features, raw_orient, conf, dim, truth_orient, truth_conf, truth_dim)
            
            loss_tot += loss.item()
            
            loss.backward()
            optimizer.step() #don't do zero grad outside
            if args.mode == 'baseline':
                optimizer.zero_grad()

            #noisy
            #if passes % 10 == 0:
            #    print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
            #    passes = 0

            passes += 1
            curr_batch += 1
        
        lr_scheduler.step()
        loss_tot /= total_num_batches
        print("=== Epoch %s completed, average loss: %s ===" % (epoch, loss_tot))
        with torch.no_grad():
            model.eval()
            val_loss_tot = 0
            for val_batch, val_labels in val_gen:
                truth_orient = val_labels['Orientation'].float().cuda()
                truth_conf = val_labels['Confidence'].long().cuda()
                truth_dim = val_labels['Dimensions'].float().cuda()

                val_batch = val_batch.float().cuda()
                [orient, conf, dim, features, raw_orient] = model(val_batch)

                orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
                dim_loss = dim_loss_func(dim, truth_dim)

                truth_conf = torch.max(truth_conf, dim=1)[1]
                conf_loss = conf_loss_func(conf, truth_conf)

                loss_theta = conf_loss + w * orient_loss
                loss = alpha * dim_loss + loss_theta
                val_loss_tot += loss.item()

            val_loss_tot /= max(1, len(val_gen))
            print("Validation loss: %s" % val_loss_tot)
            model.train()
        
        if val_loss_tot < best_loss and epoch > 20:
            best_loss = val_loss_tot
            name = model_path + 'epoch_%s_nl.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")

if __name__=='__main__':
    main()
