# Training NTM

import numpy as np
import torch
import random
import time
from tasks.copy_task import task_copy

####### Following three functions are adapted from the implemention by loudinthecloud on Github ##########
def random_seed():
    seed = int(time.time()*10000000)
    random.seed(seed)
    np.random.seed(int(seed/10000000))      # NumPy seed Range is 2**32 - 1 max
    torch.manual_seed(seed)

def clip_grads(net):    # Clipping gradients for stability
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def calc_cost(Y_out, Y, batch_size):
    y_out_binarized = Y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return cost.item()/batch_size
##########################################################################################################

def train_model(task):

    # Here, the model is optimized using BCE loss, however, it is evaluated using Number of error bits in predction and actual labels (cost)
    loss_list = []
    cost_list = []
    seq_length = []

    for batch_num, X, Y in task.get_training_data():

        task.optimizer.zero_grad()                      # Making old gradients zero before calculating the fresh ones
        task.machine.initialization(task.batch_size)    # Initializing states
        Y_out = torch.zeros(Y.shape)

        # Feeding the NTM network all the data first and then predicting output
        # by giving zero vector as input and previous read states and hidden vector
        # and thus training vector this way to give outputs matching the labels

        for i in range(X.shape[0]):
            task.machine(X[i])

        for i in range(Y.shape[0]):
            Y_out[i, :, :], _ = task.machine()

        loss = task.calc_loss(Y_out, Y)
        loss.backward()
        clip_grads(task.machine)
        task.optimizer.step()

        # The cost is the number of error bits per sequence
        cost = calc_cost(Y_out, Y, task.batch_size)

        loss_list += [loss.item()]
        cost_list += [cost]
        seq_length += [Y.shape[0]]

        print("Batch: " + str(batch_num) + "/" + str(task.num_batches) + ", Loss: " + str(loss.item()) + ", Cost: " + str(cost) + ", Sequence Length: " + str(Y.shape[0]))

def test_data(task):
    X, Y = task.get_sample_data()
    task.optimizer.zero_grad()                      # Making old gradients zero before calculating the fresh ones
    task.machine.initialization(task.batch_size)    # Initializing states
    Y_out = torch.zeros(Y.shape)

    # Feeding the NTM network all the data first and then predicting output
    # by giving zero vector as input and previous read states and hidden vector
    # and thus training vector this way to give outputs matching the labels

    for i in range(X.shape[0]):
        task.machine(X[i])

    for i in range(Y.shape[0]):
        Y_out[i, :, :], _ = task.machine()

    loss = task.calc_loss(Y_out, Y)
    loss.backward()
    clip_grads(task.machine)
    task.optimizer.step()

    # The cost is the number of error bits per sequence
    cost = calc_cost(Y_out, Y, task.batch_size)

    print("\n\nTest Data - Loss: " + str(loss.item()) + ", Cost: " + str(cost))
    
    X.squeeze(1)
    Y.squeeze(1)
    Y_out.squeeze(1)

    print("\n------Input---------\n")
    print(X.data)
    print("\n------Labels---------\n")
    print(Y.data)
    print("\n------Output---------")
    print(Y_out.data)
    print("\n")

    return loss.item(), cost, X, Y, Y_out


def main():
    # Random Seed
    random_seed()

    # Initialization of the Model
    c_task = task_copy()

    c_task.init_ntm()
    c_task.init_loss()
    c_task.init_optimizer()

    train_model(c_task)
    loss, cost, X, Y, Y_out = test_data(c_task)


if __name__ == '__main__':
    main()