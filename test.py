import sklearn
from sklearn import datasets
import numpy as np
import matplotlib as plt
from matplotlib import pyplot


#function to calculate the total loss on the dataset
def calculate_loss(model):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=np.dot(X,W1)+b1
    a1=np.tanh(z1)
    z2=np.dot(a1,W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    #calculating the loss
    correct_logprobs=-np.log(probs[range(num_examples),y])   #没看懂这里！！！！！！
    loss_sum=np.sum(correct_logprobs)
    #Adding the regulatization term to loss
    loss_sum+=reg_lambda/2*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    return 1/num_examples*loss_sum

#function to predict an output
def predict(model,x):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=np.dot(x,W1)+b1
    a1=np.tanh(z1)
    z2=np.dot(a1,W2)+b2
    exp_score=np.exp(z2)
    probs=exp_score/np.sum(exp_score,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)

def learn_model(nn_hdim,num_passes=20000,print_loss=True):
    np.random.seed(0)
    W1=np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)  #input行 nn_hdim列
    b1=np.zeros((1,nn_hdim))
    W2=np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
    b2=np.zeros((1,nn_output_dim))
    # model to return
    model={}
    for i in range(0,num_passes):
        z1=np.dot(X,W1)+b1
        a1=np.tanh(z1)
        z2=np.dot(a1,W2)+b2
        exp_score=np.exp(z2)
        probs=exp_score/np.sum(exp_score,axis=1,keepdims=True)

        #backprogation
        delta3=probs
        delta3[range(num_examples),y]-=1
        dW2=np.dot(a1.T,delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=np.dot(delta3,W2.T)*(1-np.power(a1,2))
        dW1=np.dot(np.transpose(X),delta2)
        db1=np.sum(delta2,axis=0,keepdims=True)

        #add regulatization term
        dW1+=reg_lambda*W1
        dW2+=reg_lambda*W2

        #Gradient parameters updates
        W1+=-epsilon*dW1
        b1+=-epsilon*db1
        W2+=-epsilon*dW2
        b2+=-epsilon*db2
        #assign new parameters of model
        model={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
        if print_loss and i%1000==0:
            print("loss after iteration {}:{}" .format(i,calculate_loss(model)))
    return model

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    #from matplotlib import contour
    pyplot.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    pyplot.title("hidden layer size 3")
    pyplot.show()

if __name__=='__main__':
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    # pyplot.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    # pyplot.show()

    num_examples = len(X)  # training set size
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality

    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

    # model=build_model(3,print_loss=True)
    model=learn_model(3,num_passes=40000,print_loss=True)
    print("model done")
    plot_decision_boundary(lambda x:predict(model,x))
    print("start plot pic")
