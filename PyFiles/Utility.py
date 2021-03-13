import torch


# This function is useful when monitoring gradients during training
def print_gradients(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    print(grads)


# This is also useful when monitoring training
def print_parameters(model):
    print(list(model.parameters()))
