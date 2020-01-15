import torch
'''
    In general, pruning is done in two steps
        1. Setting certain weights to 0
        2. Setting them back to 0 after every batch update
    In our case, we will use a mask, which is a binary matrix of the same size
    as our network. Its values are: 1 if the weight at that position was not pruned,
    and 0 otherwise. Pruning is then done by simply doing elementwise matrix
    multiplication between the mask and our network. Iterative pruning just implies
    updating the mask so more elements of it will be 0.
    We do step 2 in order to ensure that once a weight has been pruned,
    it will not be updated (thus making it not be 0 anymore) during backpropagation.
    This is done after every batch update, and you can either set the gradients of the weights
    to 0, thus making the loss.backward() call do nothing, or setting the magnitude to 0
    after the opt.step() call. Here, I do the second variant, since the Adam optimizer
    can set it away from 0 even if its gradient is 0, due to the momentum term in the formula.
    The torch.no_grad() call turns off the autodiff engine of Pytorch i.e. Pytorch will no longer
    keep track of operations. This is done to ensure that the stuff we do inside the block will not
    affect backpropagation. It also offers some speedup and memory efficiency.
'''

def init_mask(model):
    # Instatiate a mask as a matrix of all ones
    # of the same size as our network
    mask = [torch.ones_like(layer) for layer in model.parameters()]
    return mask

def get_total_params(model):
    # Returns total params of the model
    return sum([weights.numel() for weights in model.parameters()])

def get_sparsity(model):
    # Returns the sparsity of the model as a 
    # percentage of the total weights
    num_sparse = sum([(layer==0).sum().item()
                          for layer in model.parameters()])

    return float(num_sparse)/get_total_params(model)

def update_mask(model, mask, rate):
    '''
    Prune parameters of the network according to lowest magnitude.
    model: a pytorch model,
    mask: a binary mask of same size as our model,
    rate: is the number of remaining weights, in percentages, to prune
     i.e. on an unpruned network, update_mask(model, mask, 0.2) will prune 20% 
     of the weights. On a network that is already pruned and has, say, 80% unpruned
     weights, calling update_mask(model, mask, 0.2) will prune 16% of the weights 
     (0.8*0.2=0.16).
    '''
    with torch.no_grad():
        for layer, layer_mask in zip(model.parameters(), mask):
            # Flatten layer
            flat_layer = layer.view(-1)
            # Get indices of sorted weights
            indices = flat_layer.abs().argsort(descending=False)

            # Calculate how many weights we still need to pruen
            num_pruned = (layer_mask==0).sum().item()
            num_unpruned = layer_mask.numel() - num_pruned
            to_prune = num_pruned + int(rate*num_unpruned)
            
            # Inside the mask, set the indices of the smallest elements to 0
            indices = indices[:to_prune]
            mask = layer_mask.view(-1).clone()
            mask[indices] = 0
            # Update the mask and prune the elements by multilpying
            # the mask with the weight matrix
            layer_mask.data = mask.view_as(layer_mask)
            layer.data = layer*layer_mask

def apply_mask(model, mask):
    '''
        Apply the weights to the mask. This should be called after
        every opt.step() call in order to ensure that the weights that
        have been pruned stay at 0. 
    '''
    with torch.no_grad():
        for weights, layer_mask in zip(model.parameters(), mask):
            weights.data = weights.data*layer_mask