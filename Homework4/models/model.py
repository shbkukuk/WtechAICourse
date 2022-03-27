from cmath import cos
import numpy as np 
import matplotlib.pyplot as plt

#matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class neural_network():

    def initialize_parameters(self,n_x,n_h,n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
     
        np.random.seed(1)
        W1 = np.random.rand(n_h,n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.rand(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))

        parameters = {'W1':W1,
                      'W2':W2,
                      'b1':b1,
                      'b2':b2}

        return parameters
    
    def linear_forward(self,A,W,b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
    
        Z = np.dot(W,A)+b
        cache = (A,W,b)

        return Z,cache
    
    def linear_activition_forward(self,A_prev,W,b,activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                    stored for computing the backward pass efficiently
        """

        if activation == 'relu': 
            Z, linear_cache = neural_network().linear_forward(A_prev,W,b)
            A, activation_cache = np.maximum(0,Z), Z

        elif activation == 'sigmoid':
            Z, linear_cache = neural_network().linear_forward(A_prev,W,b)
            A, activation_cache = 1/(1+np.exp(-Z)), Z
        
        cache = (linear_cache,activation_cache)

        return A, cache

    def L_model_forward(self,X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2 
    
        for l in range(1, L):
            A_prev = A 
        
            A,cache = neural_network().linear_activition_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
            caches.append(cache)
         
        AL,cache = neural_network().linear_activition_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
        caches.append(cache)
          
        return AL, caches
    
    def compute_cost(self,AL,Y,loss_function):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
        loss_function -- mean square loss cost or root mean square cost
        
        Returns:
        cost -- mean square loss(mse) cost or root mean square cost(rmse)
        """

        m = Y.shape[1]

        if loss_function == 'mse':
            cost = (1/m)*np.sum(np.power((AL-Y),2))
        elif loss_function == 'rmse':
            cost = np.sqrt((1/m)*np.sum(np.power((AL-Y),2)))
        return cost 
    
    def linear_backward(self,dZ,cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        A_prev, W,b = cache
        m = A_prev.shape[1]

        dW = (1/m)*(np.dot(dZ,A_prev.T))
        db = np.sum(dZ,keepdims=True,axis=1)*(1/m)
        dA_prev = np.dot(W.T,dZ) 

        return dA_prev,dW,db

    def linear_activation_backward(self,dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = dA * np.where(activation_cache>0,1,0)
            dA_prev,dW,db = neural_network().linear_backward(dZ, linear_cache)
        
        elif activation == "sigmoid":
            dZ = dA * (1-1/(1+np.exp(-activation_cache)))*(1/(1+np.exp(-activation_cache)))
            dA_prev,dW,db = neural_network().linear_backward(dZ, linear_cache)
            
        return dA_prev, dW, db

    def update_parameters(self,params, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        params -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """

        parameters = params.copy()
        L = len(parameters) // 2 
        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - (learning_rate*grads['dW'+str(l+1)])
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - (learning_rate*grads['db'+str(l+1)])
         
        return parameters

    def fit(self,X, Y, layers_dims, learning_rate , num_iterations, print_cost=True):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector 
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations 
        
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """
    
        np.random.seed(1)
        grads = {}
        costs = []                         
        m = X.shape[1]                           
        (n_x, n_h, n_y) = layers_dims
        parameters = neural_network().initialize_parameters(n_x,n_h,n_y)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        print(W1.shape)
        print(W2.shape)

        for i in range(0, num_iterations):
            A1, cache1 = neural_network().linear_activition_forward(X,W1,b1,'relu')
            A2,cache2 = neural_network().linear_activition_forward(A1,W2,b2,'sigmoid')    
           
            cost = self.compute_cost(A2,Y,loss_function='mse')
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
         
            dA1, dW2, db2 = neural_network().linear_activation_backward(dA2,cache2,'sigmoid')
            dA0, dW1, db1 = neural_network().linear_activation_backward(dA1,cache1,'relu')
     
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2

            parameters = neural_network().update_parameters(parameters,grads,learning_rate)

            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs

    def plot_costs(self,costs, learning_rate=0.0075):

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    def predict(self,X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 
        p = np.zeros((1,m))
        
        probas, caches = neural_network().L_model_forward(X, parameters)

        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
       
        print ("predictions: " + str(p))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p
