import numpy as np
import timeit  # Calculating the runtime

# Defining the architecture of the neural network
nn_in = training_cols
nn_h1 = 20
nn_h2 = 20
nn_out = len(classes)

# Setting the learning rate
lr = 0.1

# Weight adjustment
wa = -0.1+(0.1+0.1)

# Number of outputs
total_output = len(training_labels)

# Constructing the matrices
x_in = np.zeros((nn_in))
w_h1 = wa*np.random.rand(nn_h1, nn_in)
b_h1 = wa*np.random.rand(nn_h1)
w_h2 = wa*np.random.rand(nn_h2, nn_h1)
b_h2 = wa*np.random.rand(nn_h2)
w_out = wa*np.random.rand(nn_out, nn_h2)
b_out = wa*np.random.rand(nn_out)
d_out = np.zeros((nn_out))

start = timeit.default_timer()

# Stores the errors and iteration number
# This denotes the number of training batches to be conducted, this is the maximum
batches = 10000
error_array = np.zeros((batches, 2))
total_err = np.zeros((batches, 1))
prev_w_h1 = wa*np.zeros((nn_h1, nn_in))
prev_w_h2 = wa*np.zeros((nn_h2, nn_h1))
prev_w_out = wa*np.zeros((nn_out, nn_h2))
# Training and Validation phase
for batch in range(batches):
    p = np.random.permutation(total_output)  # Shuffling the dataset

    # Iterates over the total number of outputs, taking in one set of inputs at a time
    for n in range(total_output):
        idx = p[n]  # Stores the shuffled index
        x_in = input[idx]
        d_out = output[idx]

        v_h1 = w_h1@x_in + b_h1
        y_h1 = 1/(1+np.exp(-v_h1))

        v_h2 = w_h2@y_h1 + b_h2
        y_h2 = 1/(1+np.exp(-v_h2))

        v_out = w_out@y_h2 + b_out
        out = 1/(1+np.exp(-v_out))

        err = d_out - out

        # compute gradient in output layer
        delta_out = err*out*(1 - out)

        # compute gradient in hidden layer 2
        delta_h2 = y_h2*(1-y_h2)*(delta_out@w_out)

        # compute gradient in hidden layer 1
        delta_h1 = (y_h1*(1-y_h1)*(delta_h2@w_h2))

        # Adjust the shape of the gradients
        delta_out_w = delta_out.reshape(-1, 1)
        delta_h2_w = delta_h2.reshape(-1, 1)
        delta_h1_w = delta_h1.reshape(-1, 1)

        # update weights and biases in output layer
        w_out = w_out + lr*delta_out_w*y_h2
        b_out = b_out + lr*delta_out

        # update weights and biases in hidden layer 2
        w_h2 = w_h2 + lr*delta_h2_w*y_h1
        b_h2 = b_h2 + lr*delta_h2

        # update weights and biases in hidden layer 1
        w_h1 = w_h1 + lr*delta_h1_w*x_in
        b_h1 = b_h1 + lr*delta_h1

    # Testing the weights on the validation set
    for q in range(len(validation_labels)):
        x_in = val_input[q]
        d_out = val_output[q]
        # hidden layer 1
        v_h1 = w_h1@x_in + b_h1
        y_h1 = 1./(1+np.exp(-v_h1))
        # hidden layer 2
        v_h2 = w_h2@y_h1 + b_h2
        y_h2 = 1./(1+np.exp(-v_h2))
        # output layer
        v_out = w_out@y_h2 + b_out
        out = 1./(1+np.exp(-v_out))
        err = d_out - out

    total_err[batch] = sum(0.5*err*err)
    average_err = sum(total_err)/np.count_nonzero(total_err)
    print('Iteration: [{}/{}] Error: {:.8f} Average Error: {:.8f}'.format(batch, batches, total_err[batch].item(),
                                                                          average_err.item()))
    error_array[batch, 0] = batch
    error_array[batch, 1] = average_err.item()
    if average_err < 0.01:
        error_array = error_array[:batch+1]
        break

stop = timeit.default_timer()

print('Time: ', stop - start)
