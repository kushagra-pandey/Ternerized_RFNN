import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import argparse

import model as M
import util as U
from SRFutil import *

def ParseArgs():
    parser = argparse.ArgumentParser(description='Ternary-Weights-Network Pytorch MNIST Example.')
    parser.add_argument('--batch-size',type=int,default=100,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=100,metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs',type=int,default=20,metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval',type=int,default=100,metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():
    args = ParseArgs()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True,**kwargs)
    bases_L1 = 10
    sigma_L1 = 1.5
    bases_L2 = 6
    sigma_L2 = 1
    bases_L3 = 6
    sigma_L3 = 1

    basis_L1 = init_basis_hermite(sigma_L1,bases_L1,5)
    basis_L2 = init_basis_hermite(sigma_L2,bases_L2,3)
    basis_L3 = init_basis_hermite(sigma_L3,bases_L3,3)

    alphas_L1 = init_alphas(64,1,bases_L1)
    alphas_L2 = init_alphas(64,64,bases_L2)
    alphas_L3 = init_alphas(64,64,bases_L3)

    w_L1 = torch.sum( alphas_L1[:,:,:,None,None] * basis_L1[None,None,:,:,:], 2)
    w_L2 = torch.sum( alphas_L2[:,:,:,None,None] * basis_L2[None,None,:,:,:], 2)
    w_L3 = torch.sum( alphas_L3[:,:,:,None,None] * basis_L3[None,None,:,:,:], 2)
    w_L4 = init_weights((3136, 10))
    #model = M.LeNet5(w_L1, w_L2, w_L3, w_L4)
    model = M.Net()
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate,rho=0.95)
    ternarize_op = U.TernarizeOp(model)
    
    best_acc = 0.0 
    for epoch_index in range(1,args.epochs+1):
        adjust_learning_rate(learning_rate,optimizer,epoch_index,args.lr_epochs)
        train(args,epoch_index,train_loader,model,optimizer,criterion,ternarize_op)
        acc = test(args,model,test_loader,criterion,ternarize_op)
        if acc > best_acc:
            best_acc = acc
            ternarize_op.Ternarization()
            U.save_model(model,best_acc)
            ternarize_op.Restore()

def train(args,epoch_index,train_loader,model,optimizer,criterion,ternarize_op):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        #if batch_idx>100:
            #break
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()
        
        ternarize_op.Ternarization()

        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        
        ternarize_op.Restore()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args,model,test_loader,criterion,ternarize_op):
    model.eval()
    test_loss = 0
    correct = 0

    ternarize_op.Ternarization()
    for i, (data,target) in enumerate(test_loader):
        #if i>50:
            #break
        data,target = Variable(data),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return acc
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    main()
