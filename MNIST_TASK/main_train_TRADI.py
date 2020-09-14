# imports
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
import time
import cv2
import torchvision.transforms as transforms
from data_loader import DatasetFromFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import utils_functions
from net1 import ThreehiddedLayersNet_fixed2, ThreehiddedLayersNet_fixed
#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from metric_OOD import  eval_ood_measure

try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")
# Parameters
N=128 # Batchsize
log_interval =200 # ploting frequence
num_epoch=30 # Num epoch
size_data_train = 60000

lr=1e-3
reg_w_decay=1e-2
args_dico={}
args_dico['name']='reslt_3hidden_layers_old_test'
args_dico['save_dir']='./reslt_3hidden_layers_test_2'
args_dico2=args_dico.copy()

seed=2

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

try:
    os.mkdir(args_dico['save_dir'])
except OSError:
    print ("Creation of the directory %s failed" % args_dico['save_dir'])
else:
    print ("Successfully created the directory %s " % args_dico['save_dir'])

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.MNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.MNIST('./data',
    download=True,
    train=False,
    transform=transform)

#parameter of the DNN
D_in=784
H1=200 # number of neurons first hidden layer
H2=200 # number of neurons second hidden layer
H3=200 # number of neurons third hidden layer
nb_class=10 # number of classes
nb_class=10
nb_models=20

# dataloaders
trainloader_mnist = torch.utils.data.DataLoader(trainset, batch_size=N,
                                        shuffle=True, num_workers=2)


testloader_mnist = torch.utils.data.DataLoader(testset, batch_size=N,
                                        shuffle=False, num_workers=2)


folders_notMNIST = os.listdir('./notMNIST_small')

NotMNIST_x_list = []
NotMNIST_y_list_onehot = []
NotMNIST_y_list = []
for idx, folder in enumerate(folders_notMNIST):
    files_notMNIST = os.listdir('./notMNIST_small/' + folder)

    for file in files_notMNIST:
        img_NotMNIST = cv2.imread('./notMNIST_small/' + folder + '/' + file, 0)

        NotMNIST_x_list.append(img_NotMNIST)

        label_temp = np.zeros([nb_class])
        label_temp[idx] = 1

        NotMNIST_y_list_onehot.append(label_temp)
        NotMNIST_y_list.append(idx)

NotMNIST_x = np.stack(NotMNIST_x_list, axis=0)
NotMNIST_y = np.asarray(np.stack(NotMNIST_y_list, axis=0),np.long)

print("NotMNIST X shape: " + str(NotMNIST_x.shape))
print("NotMNIST Y shape: " + str(NotMNIST_y.shape))

trainloader_NOTmnist = torch.utils.data.DataLoader(DatasetFromFolder(NotMNIST_x, NotMNIST_y,transform=transform, phase='test'), batch_size=N, shuffle=False, num_workers=2)



net = ThreehiddedLayersNet_fixed(D_in, H1,H2,H3,nb_class,0.1)


net_mu = ThreehiddedLayersNet_fixed(D_in, H1,H2,H3,nb_class,0.1)

net_var = ThreehiddedLayersNet_fixed(D_in, H1,H2,H3,nb_class,0.1)


net_mu = utils_functions.prepare_mu_model(net, net_mu)
net_var = utils_functions.prepare_var_model(net, net_var)

net= net.to(device)
net_mu= net_mu.to(device)
net_var= net_var.to(device)



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=reg_w_decay)

lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)



dataiter = iter(trainloader_mnist)






def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




def train( model,model_mu,model_var, device, train_loader, optimizer, epoch,Pk_fc_mu,Pk_fc_var,log_interval=50):

    model.train()
    running_loss = 0.0


    lr=get_lr(optimizer)
    for batch_idx, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        inputs = inputs.view(inputs.shape[0], -1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Update weights using gradient descent
        n_iter = epoch * len(train_loader) + batch_idx
        with torch.no_grad():
            Pk_fc_mu, Pk_fc_var=utils_functions.track_mean_var(model,model_mu,model_var,optimizer,n_iter,len(train_loader),Pk_fc_mu, Pk_fc_var)







        if batch_idx % log_interval == 0:
            #print('Epoch:', epoch, 'LR:', get_lr(optimizer))
            pred = outputs.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
           # print([K_kfc1.item(),K_kfc2.item(),K_kfc3.item(),K_kfc4.item()])
            print('Train Epoch: {}, LR: {}, [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, get_lr(optimizer),batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # ...log the running loss


    #lr_scheduler.step(running_loss)
    lr_scheduler.step()
    return Pk_fc_mu,Pk_fc_var




def test( model, device,optimizer, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            inputs = inputs.view(inputs.shape[0], -1)
            output = model(inputs)
            test_loss += criterion(output, target)# F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('CHECK', len(test_loader))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    return correct




def test_uncertainty( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    BS=0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):

            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            #inputs = inputs.view(inputs.shape[0], -1)
            output = model(inputs)
            if batch_idx ==0:
                output_concat = output.clone()
                target_concat = target.clone()
            #print('output_proba',output_proba)
            else:
                output_concat=torch.cat((output_concat, output), 0)
                target_concat=torch.cat((target_concat, target), 0)



    return output_concat , target_concat




Pk_fc_mu,Pk_fc_var=0,0 # default initial kalman gain


best_correct = 0
for epoch in range(num_epoch):  # loop over the dataset multiple times
    # training
    Pk_fc_mu, Pk_fc_var=train(net,net_mu,net_var, device, trainloader_mnist, optimizer, epoch,Pk_fc_mu,Pk_fc_var,log_interval)

    #print("KALMAN gain :K_mu / K_var",Pk_fc_mu,'/',Pk_fc_var)
    print("KALMAN gain :K_mu / K_var", Pk_fc_mu, '/', Pk_fc_var, "MEAN:  / var =>", net_var.fc1.weight.mean())

    test_loss=test(net, device,optimizer, testloader_mnist)

    if test_loss > best_correct: # save just the best on the test set

        print('SAVE BEST MODEL',best_correct,test_loss)
        utils_functions.save_checkpoint_kalman(net,net_mu,net_var, optimizer, epoch, 0,  args=None,args_dico=args_dico)

        best_correct=test_loss






net, net_mu, net_var=utils_functions.load_checkpoint_kalman(net,net_mu,net_var, optimizer, args_dico['save_dir'], args_dico['name']) # A decommenter

print('The training is finished !')

print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')

print('EVALUTION NOTMNIST (OOD)')

output=torch.zeros(len(trainloader_NOTmnist.dataset),nb_class).to(device)

m=torch.nn.Softmax(dim=1)


for step in range(nb_models):
    #print('--------------UPDATING BN-----------')
    net4 = ThreehiddedLayersNet_fixed2(D_in, H1, H2, H3, nb_class, 0.1)
    net4 = net4.to(device)
    net4 = utils_functions.load_fullnet_kalman(net, net_mu, net_var, net4, sigma=1, dimfeature=10)
    utils_functions.bn_update(trainloader_mnist, net4, 'cuda')
    #print('--------------Finished UPDATING BN-----------')
    output_temp, target = test_uncertainty(net4, device, trainloader_NOTmnist)
    output += output_temp


    # EVALUTION

m = torch.nn.Softmax(dim=1)
output = output / nb_models

NLL = criterion(output, target).item()

output_proba = m(output)


pred = output.argmax(dim=1, keepdim=True)

target_onehot = torch.nn.functional.one_hot(target, nb_class)

BS = ((target_onehot.type(torch.cuda.FloatTensor) - output_proba) * (
        target_onehot.type(torch.cuda.FloatTensor) - output_proba)).sum().item()

correct = pred.eq(target.view_as(pred)).sum().item()

scores, _ = output_proba.max(1)
scores = scores.view(-1)
scores_notmnits=scores.clone()
labels0 = target.view(-1)
pred0 = pred.view(-1)


output_proba_score0 = output_proba
output_proba_score0 = output_proba_score0.clone().cpu().data.numpy()

output_proba_score_NOTMNIST = np.amax(output_proba_score0, 1, keepdims=True)





# We save histrogram so check the ID/OOD score distribution
_ = plt.hist(output_proba_score_NOTMNIST,
             bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")

plt.savefig('Histogram_NOT_MNIST_tradi.png')
plt.close()



print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')


print('EVALUTION MNIST (ID)')



output=torch.zeros(len(testloader_mnist.dataset),nb_class).to(device)
m=torch.nn.Softmax(dim=1)



for step in range(nb_models):
    # GENERATION

    #print('--------------UPDATING BN-----------')
    net4=ThreehiddedLayersNet_fixed2(D_in, H1, H2, H3, nb_class, 0.1)
    net4 = net4.to(device)
    net4=utils_functions.load_fullnet_kalman(net, net_mu,net_var,net4, sigma=1, dimfeature=10)
    utils_functions.bn_update(trainloader_mnist, net4, 'cuda')
    #print('--------------Finished UPDATING BN-----------')
    output_temp, target = test_uncertainty(net4, device, testloader_mnist)
    output += output_temp




# EVALUTION



m=torch.nn.Softmax(dim=1)
output=output/nb_models


NLL = criterion(output, target).item()

output_proba = m(output)


pred = output.argmax(dim=1, keepdim=True)

target_onehot = torch.nn.functional.one_hot(target, nb_class)

BS = ((target_onehot.type(torch.cuda.FloatTensor) - output_proba) * (
            target_onehot.type(torch.cuda.FloatTensor) - output_proba)).sum().item()

correct = pred.eq(target.view_as(pred)).sum().item()


scores, _ = output_proba.max(1)
scores = scores.view(-1)
labels0 = target.view(-1)
pred0=pred.view(-1)
# EVALUATION OF THE OOD
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
conf=(torch.cat((scores_notmnits, scores), 0) )
label=(torch.cat((10 * torch.ones_like(scores_notmnits.view(-1)).long(), labels0.long()), 0) )
pred=(torch.cat((0 * torch.ones_like(scores_notmnits.view(-1)).long(), pred0.long()), 0) )
res = eval_ood_measure(conf, label,pred)
auroc, aupr, fpr, ece = res
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


output_proba_score0=output_proba
output_proba_score0=output_proba_score0.clone().cpu().data.numpy()

output_proba_score_MNIST=np.amax(output_proba_score0,1,keepdims=True)
print('correct = ',correct/ len(testloader_mnist.dataset))
print('error = ',1-correct/ len(testloader_mnist.dataset))
print('BS = ',BS/ len(testloader_mnist.dataset))
print('NLL = ',NLL)


# We save histrogram so check the ID/OOD score distribution
_ = plt.hist(output_proba_score_MNIST, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.title("Histogram with 'auto' bins")



plt.savefig('Histogram_MNIST.png')
plt.close()



print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')


print("mean auroc = ", auroc, "mean aupr = ", aupr, " mean fpr = ", fpr,  " mean ECE = ", ece)
