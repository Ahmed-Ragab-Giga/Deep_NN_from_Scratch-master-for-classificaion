'''
    am I have to call seed every time !???
    I really don't understand how to calculate the backpropagation shit ............

    Q's:
    1 - why do we need small initial weights !???
    2- why my random numbers aren't same as them in Jypter not book , I guess I have a series problem

'''

import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from testCases_v4 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


def initialize_parameters_deep(number_of_layers , layers_structure):
    '''
    :param L: is the number of layers in the deep NNW without the input layer
    :return parameters:
    '''

    parameters=dict()
    for i in range(1 , number_of_layers):
        # build the weights and the biases
        cl=layers_structure[i]
        pl=layers_structure[i-1]
        w=np.random.rand(cl , pl)*0.1  # need small initial weights ! why !???
        b=np.zeros((cl , 1))
        parameters["W"+str(i)]=w
        parameters["B" + str(i)] = b

    return parameters

# they gave us them implemented try to use them if some issue get arrised

def RELU(Z):
    return np.maximum(0 , Z)

def SIGMOID(Z):
    return 1/(1+np.exp(-Z))

def linear_forward(A , W , B):
    '''
        move one layer forward with all the examples
        Q/ he needs me to return the cashe ? why !???
    '''
    Z=np.dot(W , A)+B
    return Z , (A , W , B)

def linear_activation_forward(A_prev , W , B , activation):
    '''
     cache is the parameters need to get me current value
     move me one layer forward
    :param activation: tell me which function to choose to calculate the activation of my Z
    :return:
    '''

    Z , liner_cache=linear_forward(A_prev , W , B)
    if activation=="relu" :
        A_cur =RELU(Z)  # activation cache have the Z
        return A_cur , ( (A_prev , W , B) , Z )
    else:
        A_cur=SIGMOID(Z)
        return A_cur , ( (A_prev , W , B) , Z )

def forward_propogation(X , parameters , number_of_layers):
    '''
        ********************************* A_cur copy !???  ***********************
        we need some cach here for backward propagation
        cache have ((A , W , B) , Z)  # all components of the forward step
    '''
    A_cur=copy.copy(X)
    caches=[]
    caches.append((0 , 0 , 0) , (0))     # to keep the index convienent.
    for i in range(1 , number_of_layers):
        W=parameters["W" + str(i)]
        B=parameters["b" + str(i)]
        A_cur , cache=linear_activation_forward(A_cur , W , B , "relu")
        caches.append(cache)

    # output layer
    W = parameters["W" + str(number_of_layers)]
    B = parameters["b" + str(number_of_layers)]
    A_cur , cache=linear_activation_forward(A_cur , W , B , "sigmoid")

    caches.append(cache)

    return A_cur  , caches # out put probability of all the input examples as a "row"

def compute_cost(A , Y):
    rows=Y.shape[1]
    cost=(-1/rows)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))   # - cost , + is max liklihood
    cost=np.squeeze(cost)
    return cost


# back propagation part which I don't understand
def linear_backward(dz , cache): # it's a block in the backward propagation process
    #inputs were given for this layer to get it's activated result
    A_prv , W , B  = cache  # no need for Z
    rows = A_prv.shape[1]

    # calculate the derivatives
    dw=(1/rows)*np.dot(dz , np.transpose(A_prv))
    db=(1/rows)*np.sum(dz , axis=1 , keepdims=True)  # I don't understand this !???*************************************** and I didn't implement squeze !???
    da_prv=np.transpose(W).dot(dz)

    return da_prv , dw , db   # block of backward propogation


def linear_activation_backward(dA , cache , activation):
    '''
        this move me from layer to the one before it
    '''
    linea_cache , activation_cache=cache
    if activation=="relu":
        dz=relu_backward(dA , activation_cache)
        da_prv , dw , db =linear_backward(dz , linea_cache)
        return da_prv , dw , db
    else:
        dz=sigmoid_backward(dA , activation_cache)
        da_prv , dw , db =linear_backward(dz , linea_cache)
        return da_prv , dw , db



def backward_propogation(Y , AL , caches , number_of_layers):
    # AL is the activation of the last layer in the DNNW
    dAL=-(np.divide(Y , AL)-np.divide(1-Y , 1-AL))
    grads=dict()

    # last layer
    da_prev , dw , db=linear_activation_backward(dAL , caches[number_of_layers] , "sigmoid")  # layer L
    # cache the gradients to update the weights
    grads["DA_prev" + str(number_of_layers)] = da_prev
    grads["DW"+str(number_of_layers)]=dw
    grads["DB"+str(number_of_layers)]=db

    for i in range(number_of_layers-1 , 0 , -1):
        da_prev , dw , db=linear_activation_backward(da_prev , caches[i] , "relu")

        # cache the gradients to update the weights
        grads["DA_prev" + str(i)] = da_prev
        grads["DW" + str(i)] = dw
        grads["DB" + str(i)] = db

    return grads
def update_parameters(parameters , grads , learning_rate , number_of_layers):
    for i in range(1 , number_of_layers+1):
        parameters["W" + str(i)] =parameters["W"+str(i)]-learning_rate*grads["DW"+str(i)]
        parameters["B" + str(i)] = parameters["B" + str(i)] - learning_rate * grads["DB" + str(i)]

def model_deep_NNW(number_of_iterations , learnign_rate):
    for i in range(0 , number_of_iterations):
        pass
    pass

# hyper parameters
number_of_layers = 2  # excluding the input layer
number_of_iterations = 1500
learning_rate = 0.001
layers_structure = [2, 2, 3]  # input , deep , output  # number of units in each layer
# end


#parameters=initialize_parameters_deep(number_of_layers , layers_structure)
#model_deep_NNW(number_of_iterations , learning_rate)