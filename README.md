# Understanding-and-Implementing-the-Activation-Function
##       Explain the Activation Function, including its equation and graph.

Activation functions are functions used in a neural network to compute the weighted sum of inputs and biases, which is in turn used to decide whether a neuron can be activated or not. It manipulates the presented data and produces an output for the neural network that contains the parameters in the data. The activation functions are also referred to as transfer functions in some literature. These can either be linear or nonlinear depending on the function they represent and are used to control the output of neural networks across different domains.

### Different types of Activation Functions

# 1. Sigmoid activation function
## It is defined as: sigmoid(x) = 1 / (1 + exp(-x)).
The sigmoid function always returns a value between 0 and 1.

## Code
  import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Test
x_values = np.linspace(-10, 10, 100)  # create 100 points between -10 and 10
y_values = sigmoid(x_values)

Flot
plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("INPUT")
plt.ylabel("OUTPUT")
plt.grid(True)
plt.show()



              




## Graph
![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/69559c75-9cd5-42ce-af57-61ea1b329061)

### 1. It is commonly used for models where we have to predict the probability as an output. Since the probability of anything exists only between the range of 0 and 1, sigmoid is the right choice because of its range. 
### 2. The function is differentiable and provides a smooth gradient, i.e., preventing jumps in output values. This is represented by an S-shape of the sigmoid activation function. 

# 2.  Tanh activation function
## It is define as f(x) = tanh(x) = 2/(1+e^(-2x))-1
## tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).

# Code:
import numpy as np
import matplotlib.pyplot as plt

tanh function
def tanh_function(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 400)

Compute tanh values for each x
y = tanh_function(x)

Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='tanh(x)', color='green')
plt.title('Tangent Activation Function (tanh)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

Setting the x and y axis limits
plt.axhline(y=0, color='black',linewidth=0.5)
plt.axvline(x=0, color='black',linewidth=0.5)
plt.legend()
plt.show()

## Graph
![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/bcd6d7d4-468c-4a40-bab9-9b716665defc)
### 1. The output of the tanh activation function is Zero centered; hence we can easily map the output values as strongly negative, neutral, or strongly positive.
### 2. Usually used in hidden layers of a neural network as its values lie between -1 to; therefore, the mean for the hidden layer comes out to be 0 or very close to it. It helps in centering the data and makes learning for the next layer much easier.

# 3. ReLU activation function

##  A(x) = max(0,x). 
 It gives an output x if x is positive and 0 otherwise.

## Code:
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate data
x = np.linspace(-10, 10, 400)
y = relu(x)


plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU Function', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid()
plt.legend()
plt.show()

## Graph
![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/c3eb7c5d-cd29-4db0-911a-86e5931ad6a9)
### 1. Since only a certain number of neurons are activated, the ReLU function is far more computationally efficient when compared to the sigmoid and tanh functions.
### 2. ReLU accelerates the convergence of gradient descent towards the global minimum of the loss function due to its linear, non-saturating property.

# 4. Leaky ReLU activation function
##   f(x)=max(0.01*x , x)
## Range  (-inf to inf)

# Code
  import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


x = np.linspace(-10, 10, 400)
y = leaky_relu(x)


plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.legend()
plt.show()

# Graph
![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/b7e5907f-7158-4404-b9ec-b40915ad6e13)
### 1. The predictions may not be consistent for negative input values. 
### 2. The gradient for negative values is a small value that makes the learning of model parameters time-consuming.


# 5. Softmax activation function 
## The softmax computed as exp(x) / sum(exp(x)).
## Range [0, 1] and sum to 1.

# Code
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)


labels = ['x1', 'x2', 'x3']
plt.bar(labels, outputs)
plt.ylabel('Probability')
plt.title('Softmax Activation Output')
plt.show()

# Graph
![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/ec28c430-4ddf-4ef7-92ee-8be581c80497)

### Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In a general way, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will help determine the target class for the given inputs.

## Activation functions are an essential component of neural networks as they introduce non-linearity into the model, allowing it to learn complex patterns and relationships in the data. Here are some advantages and disadvantages of activation functions in neural networks:

## Advantages:

1. Activation functions introduce non-linearity into the model, enabling the network to learn complex patterns and relationships in the data.
2. Activation functions help in gradient propagation during backpropagation, allowing the model to learn effectively and converge to a solution.
3. By introducing non-linearity and enabling the model to learn complex patterns, activation functions can improve the performance of the neural network on various tasks.
4. There are various activation functions to choose from, each with its own advantages and disadvantages, allowing for flexibility in designing and optimizing neural network architectures.
 
## Disadvantages:

1. Some activation functions, such as sigmoid and tanh, can suffer from vanishing or exploding gradient problems, making it difficult for the model to learn effectively.
2. Some activation functions have a limited range of output values, which may constrain the model's ability to learn complex patterns and relationships in the data.
3. Certain activation functions, such as the exponential functions used in the softmax activation, can be computationally expensive and slow down the training process.
4. Choosing the right activation function for a neural network can be challenging, as different activation functions may perform better or worse depending on the specific task and data.

## Discuss the impact of the Activation function on gradient descent and the problem of vanishing gradients 

### The choice of activation function in a neural network can have a significant impact on the training process, particularly on the optimization technique known as gradient descent. Here's how the activation function can affect gradient descent and the issue of vanishing gradients:

Impact on Gradient Descent: Gradient descent is a popular optimization algorithm used to minimize the loss function in neural networks by iteratively updating the model parameters in the direction of the steepest descent of the loss function. The gradient of the loss function with respect to the model parameters is computed during backpropagation, and this gradient is used to update the parameters.
The choice of activation function can impact the smoothness and continuity of the loss function, which in turn affects the gradients computed during backpropagation. Activation functions that are smooth and have well-defined derivatives can facilitate the gradient descent process, leading to faster convergence and better training performance. On the other hand, activation functions that are not smooth or have discontinuities can make it challenging for gradient descent to converge to an optimal solution.

![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/285f27b8-6a90-4664-b884-6deb0a6cfb13)


Problem of Vanishing Gradients: The problem of vanishing gradients occurs when the gradient of the loss function with respect to the model parameters becomes too small during backpropagation, causing the model to learn slowly or get stuck in a suboptimal solution. This issue is particularly prevalent in deep neural networks with many layers, as the gradients tend to diminish as they propagate backward through the network.
Certain activation functions, such as the sigmoid and tanh functions, are prone to causing vanishing gradients because their gradients become very close to zero for extreme input values, leading to slow learning and difficulties in training deep networks. As a result, models with deep architectures using these activation functions may struggle to learn complex patterns and relationships in the data efficiently.
To address the problem of vanishing gradients, researchers have developed alternative activation functions such as ReLU (Rectified Linear Unit) and variants like Leaky ReLU, which have shown to alleviate the vanishing gradient issue and improve the training of deep neural networks. These functions allow gradients to flow more freely during backpropagation, enabling the network to learn more effectively and converge to better solutions.


![image](https://github.com/NaimWorld77/Understanding-and-Implementing-the-Activation-Function/assets/86626098/67ef6a4b-413e-4fc8-9dc7-f6276aaad2c2)





