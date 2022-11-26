from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    使用模块化层设计的具有ReLU非线性和softmax损耗的两层全连接神经网络。
    我们假设输入维度为D，隐藏维度为H，并对C类进行分类。
    架构应该是仿射-relu仿射-softmax。
    注意，这个类不实现梯度下降；相反，它将与负责运行的单独解算器对象交互优化。
    模型的可学习参数存储在self.params中
    自己将参数名称映射到numpy数组的参数。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        给出输入数据的规模
        - hidden_dim: An integer giving the size of the hidden layer
        隐藏网络的规模
        - num_classes: An integer giving the number of classes to classify
        共有多少个类
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        标量给出权重随机初始化的标准偏差。
        - reg: Scalar giving L2 regularization strength.
        正则项
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.             
        #初始化双层网的权重和偏差。
        #权重应从以0.0为中心的高斯函数初始化，标准偏差等于weight_scale，偏差应初始化为零。
        #所有权重和偏差都应存储在self.params中。参数，第一层权重和偏置使用关键字“W1”和“b1”，
        #第二层权重和偏差使用键“W2”和“b2”。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1']=np.random.normal(loc=0,scale=weight_scale,size=(input_dim,hidden_dim))
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params['b2']=np.zeros(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.  
        #完成两层神经网络的前向传播，计算X每一个类别的分数，将他们存储在scores中        #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out1,cache1=affine_forward(X,self.params['W1'],self.params['b1'])
        out2,cache2=relu_forward(out1)
        out3,cache3=affine_forward(out2,self.params['W2'],self.params['b2'])
        loss,dx=softmax_loss(out3,y)
        loss+=0.5*self.reg*(np.sum(self.params['W1']*(self.params['W1']))+np.sum(self.params['W2']*self.params['W2']))
        scores=out3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        grads={}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.          
        dx,grads['W2'],grads['b2']=affine_backward(dx,cache3)
        dx=relu_backward(dx,cache2)
        m,grads['W1'],grads['b1']=affine_backward(dx,cache1)
        grads['W2']+=grads['W2']*self.reg
        grads['W1']+=grads['W1']*self.reg
        #待办事项：
        #实现两层网络的反向通行。将损失存储在损失变量中，梯度存储在梯度字典中。
        #使用softmax计算数据丢失，并确保grads[k]保持self.params[k]的梯度。别忘了添加L2正则化！
        #注意：为了确保您的实现与我们的匹配，并且您通过了自动化测试，
        #请确保您的L2正则化包含0.5的因子，以简化梯度的表达式。            #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
