from calendar import c
from turtle import color
import numpy as np 
import matplotlib.pyplot as plt

def generate_data(n,beta_0,beta_1):
    """
    The fuction create dataset that contains points (x,y)

    Parametres:
    n(int) : how much contain data points
    beta_0(int) : bias value
    beta_1(int) : weight  value

    Return:
    x,y : array points value
    """
    x = np.arange(n)
    e = np.random.uniform(-10,10, size=(n,))
    y = beta_0 + beta_1* x + e
    return x,y 

class linearRegession():
    '''
    This is a class for prepare data learning model that is Linear Regession
    
    Example:
        linearRegession(x,y,learning_rate,iters)
    Args:
        x (array): The arg is used for input x 
        y (array): The arg is used for outpot of eqution
        learning_rate (int):Gradient descent steps coefficient
        iters(int): How many times do it 
    
        
    '''
    def __init__(self,x,y,learning_rate,iters) :
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.iters = iters
        
    def predict_y(self,x,weight,bias):
        return x*weight+bias

    def cost_function(self,weight,bias):
        total_error = 0
        total_data = len(self.x)
        for i in range(total_data):
            total_error += (self.predict_y(self.x[i],weight,bias)-self.y[i])**2  
        return total_error / total_data

    def gradient_descent(self,weight,bias):
        weight_deriv = 0
        bias_deriv = 0
        total_data = len(self.x)
        for i in range(total_data):
            bias_deriv += -2*(self.y[i]-(bias+weight*self.x[i]))
            weight_deriv += -2*self.x[i]*(self.y[i]-(bias+weight*self.x[i]))
        weight = weight - weight_deriv*self.learning_rate
        bias = bias -bias_deriv*self.learning_rate
        return weight,bias

    def train(self,weight,bias):
        cost_history = list()
        iteration = list()
        for i in range(self.iters):
            weight , bias = self.gradient_descent(weight,bias)
            cost = self.cost_function(weight,bias)
            

            if i % 5 == 0 :
                iteration.append(i)
                cost_history.append(cost)
                print (f'iter:{i}, weight:{weight:.2f}, bias: {bias:.2f}, cost: {cost:.2f} ')
                y1 = weight*self.x + bias
                plt.title('Points')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.scatter(self.x,self.y ,color='blue')
                plt.plot(self.x,y1,color='red')
                plt.show()
        return cost_history,iteration
        


if __name__ == '__main__':
    #Create some data points
    x , y = generate_data (100,2,.4)

    #Create object of linearRegession class
    model = linearRegession(x,y,.000002,40)
    costs , iters = model.train(weight=.9,bias=3)#fit model and return cost and iters

    #Plot the MSE of own regression function during iterations.
    plt.xlabel('Training Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Cost Function')
    plt.plot(iters,costs)
    plt.show()