import os
import numpy as np
import torch


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha





# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        b = input.size(0)

        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum

        if device is not None:
            input = input.to(device)

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def track_mean_var(net1, net2,net3,optim,n_iter,lendata, PK_mu=0, PK_var=0,epoch_limit=10):
    '''for param1, param2 in zip(net1.parameters(), net2.parameters()):
        flag = [False]'''
    list_bn = list_bn_fc(net1)
    lr0 = get_lr(optim)
    #print('je passe ici')
    Q_fc1k_minus_1 = 1e-4
    Q_fc1k_minus_1_var = 5e-5
    Pk_fc1_minus = PK_mu + Q_fc1k_minus_1
    # kalman gain at step k
    K_kfc1 = Pk_fc1_minus / (Pk_fc1_minus + 1e1 )
    Pk_fc1_minus_var= PK_var + Q_fc1k_minus_1_var
    K_kfc1_var = Pk_fc1_minus_var / (Pk_fc1_minus_var + 1e1)


    state_dict_mu_kalman = net2.state_dict()
    state_dict_var_kalman = net3.state_dict()

    for name, param in net1.named_parameters():


        isbn = False
        for name_bn in list_bn:
            if name_bn in name:
                isbn = True
        if isbn:

            transformed_param=param
            transformed_param_var = param

        else:

            transformed_param = (1 - K_kfc1) * (
                    (net2.state_dict()[name]) - lr0 * param.grad) + K_kfc1 * param

            transformed_param_var = (1 - K_kfc1_var) * ((net3.state_dict()[name]) + (lr0 * param.grad) * (
                    lr0 * param.grad)) + K_kfc1_var * (param - transformed_param) * (param - transformed_param)


            # Update the parameter.
        state_dict_mu_kalman[name].copy_(transformed_param)

        state_dict_var_kalman[name].copy_(transformed_param_var)
        PK_mu = (1 - K_kfc1) * Pk_fc1_minus
        PK_var = (1 - K_kfc1_var) * Pk_fc1_minus_var
    net2.load_state_dict(state_dict_mu_kalman)
    net3.load_state_dict(state_dict_var_kalman)
    if n_iter%lendata==1 : print(K_kfc1_var)

    return PK_mu,PK_var



def list_bn_fc(model):
    list_bn=[]
    for name, module1 in model.named_modules():

         if  isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
             #transformed_param = param
             #print('bn =>',name,module1)
             list_bn.append(name)

    return list_bn


def prepare_mu_model(model,model_kalman):
    """put to zeros all the weights of the model
    Keyword arguments:
    - model (``nn.Module``): The model to save.
    """
    state_dict = model_kalman.state_dict()

    list_bn=list_bn_fc(model)


    state_dict = model_kalman.state_dict()
    for name, param in state_dict.items():
        # Transform the parameter as required.
        isbn=False
        for name_bn in list_bn:
            if name_bn in name:
                isbn=True
        if isbn:
            transformed_param = param
        else:
            transformed_param = torch.zeros_like(param)

        # Update the parameter.
        state_dict[name].copy_(transformed_param)
    model_kalman.load_state_dict(state_dict)

    return model_kalman




def prepare_var_model(model,model_kalman):
    """put to zeros all the weights of the model
    Keyword arguments:
    - model (``nn.Module``): The model to save.
    """
    list_bn=list_bn_fc(model)


    state_dict = model_kalman.state_dict()
    for name, param in state_dict.items():
        # Transform the parameter as required.
        isbn=False
        for name_bn in list_bn:
            if name_bn in name:
                isbn=True
        if isbn:
            transformed_param = param
        else:
            #print(name,param.size())
            transformed_param = torch.ones_like(param) * np.sqrt(2 / param.size()[0])

        # Update the parameter.
        state_dict[name].copy_(transformed_param)
    model_kalman.load_state_dict(state_dict)

    return model_kalman



def l2_loss(model,model_prior):
    """put to zeros all the weights of the model
    Keyword arguments:
    - model (``nn.Module``): The model to save.
    """
    loss=torch.tensor([0]).to('cuda')
    list_bn=list_bn_fc(model)
    print('---------------------------------')
    print(list_bn)
    print('---------------------------------')

    state_dict = model_prior.state_dict()
    for name, param in state_dict.items():
        # Transform the parameter as required.
        isbn=False
        for name_bn in list_bn:
            if name_bn in name:
                isbn=True
        if isbn ==False:
            loss += ((param -(model.state_dict()[name]))*(param -(model.state_dict()[name]))).sum()



    return loss

def save_checkpoint_kalman(model, model_mu, model_var, optimizer, epoch, miou, args=None,args_dico=None):
    """Saves the model in a specified directory with a specified name.save
    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".
    """
    if args!=None:
        name = args.name
        save_dir = args.save_dir
    else:
        name = args_dico['name']
        save_dir = args_dico['save_dir']

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'net_mu': model_mu,
        'net_var': model_var,
    }
    torch.save(checkpoint, model_path)
    if args != None:
        # Save arguments
        summary_filename = os.path.join(save_dir, name + '_summary.txt')
        with open(summary_filename, 'w') as summary_file:
            sorted_args = sorted(vars(args))
            summary_file.write("ARGUMENTS\n")
            for arg in sorted_args:
                arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
                summary_file.write(arg_str)

            summary_file.write("\nBEST VALIDATION\n")
            summary_file.write("Epoch: {0}\n". format(epoch))
            summary_file.write("Mean IoU: {0}\n". format(miou))



def load_checkpoint_kalman(model,model_kalman_mu,model_kalman_var, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save
    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.
    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.
    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    model_kalman_mu.load_state_dict(checkpoint['net_mu'].state_dict())
    model_kalman_var.load_state_dict(checkpoint['net_var'].state_dict())



    return model,model_kalman_mu,model_kalman_var



def load_fullnet_kalman(net1, net_mu,net_var,net4, sigma=1, dimfeature=10):

    state_dict = net4.state_dict()
    list_bn=list_bn_fc(net1)
    net4.load_state_dict(net1.state_dict(), strict=True)

    with torch.no_grad():

        for name, param in net1.named_parameters():
            # Transform the parameter as required.
            isbn = False
            for name_bn in list_bn:
                if name_bn in name:
                    isbn = True
            if not isbn:

                # print(name)
                fc_i_weight_0 = np.random.normal(0, 1, (dimfeature, 1))
                var_fc_i_weight = np.sqrt(np.maximum(net_var.state_dict()[name].clone().cpu().data.numpy(), 0))
                # var_fc_i_weight = np.sqrt(1e-4)*np.ones_like(model_var.state_dict()[name].clone().cpu().data.numpy())
                size_data_fc_i_weight = np.prod(np.shape(var_fc_i_weight))
                var_fc_i_weight = np.reshape(var_fc_i_weight, (size_data_fc_i_weight, 1))

                Z_approx_i_FC = np.multiply(var_fc_i_weight, np.cos(np.multiply(
                    np.reshape(net1.state_dict()[name].clone().cpu().data.numpy(), (size_data_fc_i_weight, 1)),
                    np.random.normal(0, sigma, (1, dimfeature))) + np.random.uniform(0, 6.28318, (
                    size_data_fc_i_weight, dimfeature))))
                # print('name',np.mean(Z_approx_i_FC)*np.sqrt(2 / (dimfeature)))
                fc_i_weight_numpy0 = np.reshape(net_mu.state_dict()[name].clone().cpu().data.numpy(),
                                                (size_data_fc_i_weight, 1)) + np.sqrt(2 / (dimfeature)) * np.matmul(Z_approx_i_FC, fc_i_weight_0)

                fc_i_weight_numpy = fc_i_weight_numpy0.astype(np.float32)
                del (fc_i_weight_numpy0)
                fc_i_weight_numpy = np.reshape(fc_i_weight_numpy,
                                               np.shape(net_mu.state_dict()[name].clone().cpu().data.numpy()))
                transformed_param = torch.tensor(fc_i_weight_numpy)
                state_dict[name].copy_(transformed_param)



        net4.load_state_dict(state_dict)
    return net4



def load_simplenet_kalman(net1, net_mu,net_var,net4, coef_sigma=1):
    '''for param1, param2 in zip(net1.parameters(), net2.parameters()):
        flag = [False]'''
    state_dict = net4.state_dict()
    list_bn=list_bn_fc(net1)
    net4.load_state_dict(net1.state_dict(), strict=True)

    with torch.no_grad():

        for name, param in net1.named_parameters():
            # Transform the parameter as required.
            isbn = False
            for name_bn in list_bn:
                if name_bn in name:
                    isbn = True
            if not isbn:

                # print(name)

                var_fc_i_weight = coef_sigma*np.sqrt(np.maximum(net_var.state_dict()[name].clone().cpu().data.numpy(), 0))
                # var_fc_i_weight = np.sqrt(1e-4)*np.ones_like(model_var.state_dict()[name].clone().cpu().data.numpy())
                size_data_fc_i_weight = np.prod(np.shape(var_fc_i_weight))
                fc_i_weight_0 = np.random.normal(0, 1, (size_data_fc_i_weight, 1))

                var_fc_i_weight = np.reshape(var_fc_i_weight, (size_data_fc_i_weight, 1))

                Z_approx_i_FC = np.multiply(var_fc_i_weight,fc_i_weight_0)
                # print('name',np.mean(Z_approx_i_FC)*np.sqrt(2 / (dimfeature)))
                fc_i_weight_numpy0 = np.reshape(net_mu.state_dict()[name].clone().cpu().data.numpy(),
                                                (size_data_fc_i_weight, 1)) + Z_approx_i_FC

                fc_i_weight_numpy = fc_i_weight_numpy0.astype(np.float32)
                del (fc_i_weight_numpy0)
                fc_i_weight_numpy = np.reshape(fc_i_weight_numpy,
                                               np.shape(net_mu.state_dict()[name].clone().cpu().data.numpy()))
                transformed_param = torch.tensor(fc_i_weight_numpy)
                state_dict[name].copy_(transformed_param)



        net4.load_state_dict(state_dict)
    return net4




