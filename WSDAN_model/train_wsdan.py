"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from optparse import OptionParser
from tensorboardX import SummaryWriter

from utils import accuracy
from models import *
from dataset import *

USE_GPU = True
device = None
best_top1_val_accuracy = 0
data_root = os.path.join('..', '..', 'dataset')  # path to images
train_file = os.path.join('..', '..', 'dataset', 'train2019.json')
val_file = os.path.join('..', '..', 'dataset', 'val2019.json')
test_file = os.path.join('..', '..', 'dataset', 'test2019.json')

# set isTest to True to run the test set (not implemented yet)
isTest = False
test_output_file_name = 'submission.csv' # submission file
if isTest:
    val_file = os.path.join('..', '..', 'dataset', 'test2018.json')


def main():
    global device, best_top1_val_accuracy
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 8)')
    parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                      help='number of epochs (default: 80)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                      help='batch size (default: 16)')
    parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                      help='load checkpoint model (default: False)')
    parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                      help='show information for each <verbose> iterations (default: 100)')

    parser.add_option('--lr', '--learning-rate', dest='lr', default=1e-3, type='float',
                      help='learning rate (default: 1e-3)')
    parser.add_option('--sf', '--save-freq', dest='save_freq', default=1, type='int',
                      help='saving frequency of .ckpt models (default: 1)')
    parser.add_option('--sd', '--save-dir', dest='save_dir', default='./saved_models',
                      help='saving directory of .ckpt models (default: ./saved_models)')
    parser.add_option('--init', '--initial-training', dest='initial_training', default=False,
                      help='train from True-beginning or False-resume training (default: False)')

    (options, args) = parser.parse_args()

    logging.basicConfig(filename='./output/training.log', filemode='a',
                        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Initialize model
    ##################################
    image_size = (512, 512)
    num_classes = 1010
    num_attentions = 32
    start_epoch = 0
    step = 0
    tbx = SummaryWriter('./output/tensorboard')

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(device)

    if options.ckpt:
        ckpt = options.ckpt

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        if not options.initial_training:
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
        best_top1_val_accuracy = checkpoint['best_top1_val_accuracy']

        # Load weights
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(options.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(options.ckpt))

    ##################################
    # Initialize saving directory
    ##################################
    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(device)
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################


    train_dataset, validate_dataset = INAT(data_root, train_file, image_size, is_train=True), \
                                      INAT(data_root, val_file, image_size, is_train=False)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                                               num_workers=options.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=options.batch_size, shuffle=False,
                                               num_workers=options.workers, pin_memory=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()

    ##################################
    # Learning rate scheduling
    ##################################
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
    logging.info('\nStart training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))

    for epoch in range(start_epoch, options.epochs):
        step = train(epoch=epoch,
              step=step,
              batch_size=options.batch_size,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              loss=loss,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose,
              tbx=tbx)
        val_loss = validate(epoch=epoch,
                            data_loader=validate_loader,
                            net=net,
                            feature_center=feature_center,
                            loss=loss,
                            save_dir=options.save_dir,
                            verbose=options.verbose,
                            tbx=tbx)
        scheduler.step()


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    batch_size = kwargs['batch_size']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    step = kwargs['step']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']
    tbx = kwargs['tbx']

    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 1e-4
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = np.array([0, 0, 0], dtype='float')  # Loss on Raw/Crop/Drop Images
    epoch_acc = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]], dtype='float')  # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    logging.info('Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, _, y, _) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)

        ##################################
        # Raw Image
        ##################################
        y_pred, feature_matrix, attention_map = net(X)

        # loss
        batch_loss = loss(y_pred, y) + l2_loss(feature_matrix, feature_center[y])
        epoch_loss[0] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Update Feature Center
        feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] = epoch_acc[0] + accuracy(y_pred, y, topk=(1, 3, 5))

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)

        # crop images forward
        y_pred, _, _ = net(crop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[1] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[1] = epoch_acc[1] + accuracy(y_pred, y, topk=(1, 3, 5))

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) <= theta_d
            drop_images = X * drop_mask.float()

        # drop images forward
        y_pred, _, _ = net(drop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[2] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[2] = epoch_acc[2] + accuracy(y_pred, y, topk=(1, 3, 5))

        # end of this batch
        batches += 1
        step += batch_size
        batch_end = time.time()

        tbx.add_scalar('train/raw_loss', epoch_loss[0] / batches, step)
        tbx.add_scalar('train/crop_loss',epoch_loss[1] / batches, step)
        tbx.add_scalar('train/drop_loss',epoch_loss[2] / batches, step)
        tbx.add_scalar('train/raw_top1_acc', epoch_acc[0, 0] / batches, step)
        tbx.add_scalar('train/raw_top2_acc', epoch_acc[0, 1] / batches, step)
        tbx.add_scalar('train/raw_top3_acc', epoch_acc[0, 2] / batches, step)
        tbx.add_scalar('train/crop_top1_acc', epoch_acc[1, 0] / batches, step)
        tbx.add_scalar('train/crop_top2_acc', epoch_acc[1, 1] / batches, step)
        tbx.add_scalar('train/crop_top3_acc', epoch_acc[1, 2] / batches, step)
        tbx.add_scalar('train/drop_top1_acc', epoch_acc[2, 0] / batches, step)
        tbx.add_scalar('train/drop_top2_acc', epoch_acc[2, 1] / batches, step)
        tbx.add_scalar('train/drop_top3_acc', epoch_acc[2, 2] / batches, step)

        if (i + 1) % verbose == 0:
            logging.info('\n\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f' %
                         (i + 1,
                          epoch_loss[0] / batches, epoch_acc[0, 0] / batches, epoch_acc[0, 1] / batches, epoch_acc[0, 2] / batches,
                          epoch_loss[1] / batches, epoch_acc[1, 0] / batches, epoch_acc[1, 1] / batches, epoch_acc[1, 2] / batches,
                          epoch_loss[2] / batches, epoch_acc[2, 0] / batches, epoch_acc[2, 1] / batches, epoch_acc[2, 2] / batches,
                          batch_end - batch_start))

    # save checkpoint model
    global best_top1_val_accuracy
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu(),
            'best_top1_val_accuracy': best_top1_val_accuracy,
            'step': step},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f'%
                 (epoch_loss[0], epoch_acc[0, 0], epoch_acc[0, 1], epoch_acc[0, 2],
                  epoch_loss[1], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
                  epoch_loss[2], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
                  end_time - start_time))
    return step


def validate(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']
    save_dir = kwargs['save_dir']
    feature_center = kwargs['feature_center']
    tbx = kwargs['tbx']

    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float') # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, _, y, _) in enumerate(data_loader):
            batch_start = time.time()

            # obtain data
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, feature_matrix, attention_map = net(X)

            ##################################
            # Object Localization and Refinement
            ##################################
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, _ = net(crop_images)

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2

            # loss
            batch_loss = loss(y_pred, y)
            epoch_loss = epoch_loss + batch_loss.item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc = epoch_acc + accuracy(y_pred, y, topk=(1, 3, 5))

            # end of this batch
            batches += 1
            batch_end = time.time()

            if (i + 1) % verbose == 0:
                logging.info('\n\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                         (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches, epoch_acc[2] / batches, batch_end - batch_start))


    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    tbx.add_scalar('val/loss', epoch_loss, epoch)
    tbx.add_scalar('val/top1_acc', epoch_acc[0], epoch)
    tbx.add_scalar('val/top2_acc', epoch_acc[1], epoch)
    tbx.add_scalar('val/top3_acc', epoch_acc[2], epoch)

    # save best model
    global best_top1_val_accuracy
    if epoch_acc[0] > best_top1_val_accuracy:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu(),
            'best_top1_val_accuracy': best_top1_val_accuracy},
            os.path.join(save_dir, 'best_top1_val_acc_%03d.ckpt' % (epoch + 1)))

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f'%
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))


    return epoch_loss


if __name__ == '__main__':
    main()
