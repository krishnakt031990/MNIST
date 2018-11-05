from __future__ import print_function
import os
import argparse
from uuid import uuid1

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter

from models.net import Net
from logger.utils import progress_bar

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument('--batch', '-b', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', '-test_b', type=int, default=512,
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--decay', '-d', default=1000, type=int,
                    help='how many epochs before the learning rate decays')
parser.add_argument('--decay_gamma', '-dg', default=0.1, type=int,
                    help='decay parameter to add this amount of decay (default: 0.1)')
parser.add_argument('--optimizer', '-opt', type=str, default='sgd',
                    help='Optimizer of choice (default: sgd)')
parser.add_argument('--momentum', '-m', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', '-wd', type=long, default=5e-4,
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--log_dir', '-log', type=str, default='./runs',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--resume', '-r', default=None, help='checkpoint file name. It ends with .ckpt')
args = parser.parse_args()

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

run_id = uuid1()
comment = "lr_" + str(args.learning_rate) + "_de_" + str(args.decay) + "_dg_" + str(args.decay_gamma) + "_e_" + str(
    args.epochs) + "_b_" + str(
    args.batch) + "_o_" + str(args.optimizer)
best_acc = 0

sw = SummaryWriter(log_dir=args.log_dir + "/" + str(run_id), comment=comment)

net = Net()

if use_cuda == 'cuda':
    ltype = torch.cuda.LongTensor
    ftype = torch.cuda.FloatTensor
    net = net.cuda()
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    ltype = torch.LongTensor
    ftype = torch.FloatTensor

dummy_input = torch.randn(1, 1, 28, 28)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_file_name = args.resume if str.endswith(args.resume) == ".ckpt" else args.resume + ".ckpt"
    checkpoint = torch.load('./checkpoint/' + ckpt_file_name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

training_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_data, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch, shuffle=True)

opt = {
    'sgd': optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay),
    'adam': optim.Adam(net.parameters(), lr=args.learning_rate)
}

criterion = nn.CrossEntropyLoss()
optimizer = opt.get((args.optimizer).lower())
scheduler = StepLR(optimizer, step_size=args.decay, gamma=args.decay_gamma)


# ================================================================== #
#                         Training                                   #
# ================================================================== #
def train(tb_step, epoch):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        tb_step += 1
        data, target = Variable(data.type(ftype)), Variable(target.type(ltype))
        outputs = net(data)
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        sw.add_scalar('loss', loss.item(), tb_step)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    sw.add_scalar('Train Accuracy', acc, epoch)

    return tb_step


# ================================================================== #
#                        Testing                                     #
# ================================================================== #
def test(epoch):
    global best_acc
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = Variable(data.type(ftype))
            label_tensor = Variable(target.type(ltype))
            outputs = net(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(label_tensor).sum().item()

    acc = 100. * correct / total
    print('Test Accuracy: %.3f' % (acc))
    sw.add_scalar('Test Accuracy', acc, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + str(run_id) + ".ckpt")
        torch.save(net, './checkpoint/' + str(run_id))
        best_acc = acc


if __name__ == '__main__':

    tb_step = 0

    for epoch in range(args.epochs):
        scheduler.step()
        tb_step = train(tb_step, epoch)
        test(epoch)

    sw.close()
