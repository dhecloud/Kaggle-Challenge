import warnings
warnings.simplefilter("ignore")
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from RadioGraphsDataset import RadioGraphsDataset
import numpy as np
import time
import argparse
import datetime
import os, sys
# import matplotlib.pyplot as plt

OUTFILE = "results"


parser = argparse.ArgumentParser(description='Pnuemonia Detection')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=150, help='number of epochs')
parser.add_argument('--test', type=bool, default=False, help='test')
parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='decay lr by 10 after _ epoches')
parser.add_argument('--no_augment', action='store_true', help='dont augment data?')
parser.add_argument('--no_validate', action='store_true', help='dont validate data when training?')
parser.add_argument('--augment_probability', type=float, default=1.0, help='augment probability')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
parser.add_argument('--persons', type=str, default=None, nargs='+', help='persons to train on')
parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--print_interval', type=int, default=100, help='print interval')
parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models. if none, it will be saved as the date and time')
parser.add_argument('--finetune', action='store_true', help='use a pretrained checkpoint')
parser.add_argument('--classifcation', action='store_true', help='use a pretrained checkpoint')


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    expr_dir = os.path.join(opt.save_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.lr_decay epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    # print("LR is " + str(lr)+ " at epoch "+ str(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def main(args):
    model = models.resnet18()

    model.float()
    model.cuda()
    model.apply(weights_init)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()

    args.augment = not args.no_augment
    args.validate = not args.no_validate
    train_dataset = RadioGraphsDataset(training = True,args=args)
    test_dataset = RadioGraphsDataset(training = False, args=args)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=args.batchSize, shuffle = True,
       num_workers=0, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=args.batchSize  ,shuffle = True,
       num_workers=0, pin_memory=False)

    current_epoch = 0
    if args.checkpoint:
        model, optimizer, current_epoch = load_checkpoint(args.checkpoint, model, optimizer)
        if args.finetune:
            current_epoch = 0

    if args.test:
        validate(val_loader, model, criterion, args)

    train_loss = []
    val_loss = []
    val_acc = []
    mean_errors = []
    best = False

    print_options(args)
    expr_dir = os.path.join(args.save_dir, args.name)

    for epoch in range(current_epoch, args.epoch):

        optimizer = adjust_learning_rate(optimizer, epoch, args)
        np.savetxt(os.path.join(expr_dir,"current_epoch.out"),[epoch], fmt='%f')
        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch, args)
        train_loss = train_loss + loss_train
        if args.validate:
            # evaluate on validation set
            loss_val, mean_error = validate(val_loader, model, criterion ,args)
            val_loss = val_loss + loss_val
            mean_errors += mean_error
            #print(mean_errors)
        state = {
            'epoch': epoch,
            'arch': "REN",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        if not os.path.isfile(os.path.join(expr_dir, 'model_best.pth.tar')):
            save_checkpoint(state, True, args)

        if (args.validate) and (epoch > 1) :
            best = (loss_val < min(val_loss[:len(val_loss)-1]))
            if best:
                print("saving best performing checkpoint on val")
                save_checkpoint(state, True, args)

        save_checkpoint(state, False, args)
    #

    expr_dir = os.path.join(args.save_dir, args.name)
    np.savetxt(os.path.join(expr_dir, "train_loss.out"),train_loss, fmt='%f')
    # save_plt(train_loss, "train_loss")
    np.savetxt(os.path.join(expr_dir, "val_loss.out"),val_loss, fmt='%f')
    # save_plt(val_loss, "val_loss")
    np.savetxt(os.path.join(expr_dir, "val_acc.out"),val_acc, fmt='%f')
    # save_plt(val_acc, "val_acc")
    np.savetxt(os.path.join(expr_dir, "mean_errors.out"),mean_errors, fmt='%f')
    # save_plt(mean_errors, "mean_errors")





def train(train_loader, model, criterion, optimizer, epoch,args):

    # switch to train mode
    model.train()
    loss_train = []
    expr_dir = os.path.join(args.save_dir, args.name)
    for i, (input, target) in enumerate(train_loader):

        stime = time.time()
        # measure data loading time
        target = target.float()
        target = target.cuda(non_blocking=False)
        input = input.float()
        input = input.cuda()
        # compute output
        output = model(input)

        loss = criterion(output, target)
        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        loss_train.append(loss.data.item())
        optimizer.step()
        np.savetxt(os.path.join(expr_dir, "_iteration_train_loss.out"), np.asarray(loss_train), fmt='%f')
        # measure elapsed time
        if i % args.print_interval == 0:
            TT = time.time() -stime
            print('epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Time: {time:.2f}\t'.format(
                   epoch, i, len(train_loader), loss=loss.item(), time= TT))

    return [np.mean(loss_train)]



def validate(val_loader, model, criterion, args):

    # switch to evaluate mode
    model.eval()

    loss_val = []
    errors = []
    with torch.no_grad():
        expr_dir = os.path.join(args.save_dir, args.name)
        for i, (input, target) in enumerate(val_loader):
            target = target.float()
            target = target.cuda(non_blocking=False)
            # compute output
            input = input.float()
            input = input.cuda()
            output = model(input)

            errors.append(compute_distance_error(output, target).item())
            loss = criterion(output, target)

            if i % args.print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(val_loader), loss=loss))
            loss_val.append(loss.data.item())
            np.savetxt(os.path.join(expr_dir, "_iteration_val_loss.out"), np.asarray(loss_val), fmt='%f')


    return [np.mean(loss_val)] , [np.mean(errors)]

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)

def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    expr_dir = os.path.join(opt.save_dir, opt.name)
    torch.save(state, os.path.join(expr_dir, filename))
    if is_best:
        torch.save(state, os.path.join(expr_dir, 'model_best.pth.tar'))

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch  = checkpoint['epoch']

    return model, optimizer, epoch



if __name__ == '__main__':
    args = parser.parse_args()
    if not args.name:
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    main(args)
