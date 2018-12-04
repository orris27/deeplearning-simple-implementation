import numpy as np


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_deriv(x):
    return 1. - tanh(x) ** 2

def sigmoid(x):
    #return np.exp(x) / (1 + np.exp(x))
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))



class NN:
    def __init__(self, layers, activation_fn="tanh"): # [1, 10, 1]

        if activation_fn == "tanh":
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation_fn == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid
        
        self.layers = layers

        self.weights = [] 
        self.biases = []
        
        for l in range(len(layers) - 1):
            self.weights.append(np.random.random([layers[l], layers[l + 1]])) # weights: [1, 10], [10, 1]
        for l in layers:
            self.biases.append(np.zeros(l))

    def loss(self, y_pred, y):
        return np.abs(y_pred - y)

    def forward(self, x):
        outputs = [x]
        for l, weight in enumerate(self.weights):
            outputs.append(np.matmul(outputs[l], weight)) # outputs[l]: [batch_size, xx]
        return outputs

    def backward(self, outputs, y): # y is the label
        batch_size = outputs[0].shape[0]

        delta_biases = [np.zeros_like(bias) for bias in self.biases]
        delta_weights = [np.zeros_like(weight) for weight in self.weights]

        for ins in range(batch_size):
            # calc errors from back to front
            # calc the errors in the last layer
            errors = []
            #errors.append(self.activation_deriv(outputs[-1][ins]) * self.loss(outputs[-1][ins], y[ins]))
            errors.append(self.activation_deriv(outputs[-1][ins]) * (y[ins] - outputs[-1][ins]))
            #errors.append(self.activation_deriv(outputs[-1][ins]) * (y[ins] - outputs[-1][ins]))

            for l, weight in enumerate(reversed(self.weights)):
                errors.append(self.activation_deriv(outputs[-(l + 2)][ins]) * np.matmul(weight, errors[l])) 
            errors = list(reversed(errors))

            # calc the delta
            for i in range(len(self.layers)): # [1, 10, 1]
                delta_biases[i] += self.learning_rate * errors[i]

            for i in range(len(self.layers) - 1):
                delta_weights[i] += self.learning_rate * np.matmul(outputs[i][ins], errors[i]) # [1, 10], O_i:[1], Err_j=Err_{j+1}:[10]

        
        tmp = 1
        for i in range(len(self.biases)):
            self.biases[i] += delta_biases[i] / batch_size

        for i in range(len(self.weights)):
            self.weights[i] += delta_weights[i] / batch_size



    def fit(self, x, y, num_epochs, learning_rate=0.1):

        self.learning_rate = learning_rate

        for epoch in range(num_epochs):
            outputs = self.forward(x) # len(outputs): 1
            self.backward(outputs, y)

            if epoch % 100 == 0:
                print("epoch {0}: loss={1}".format(epoch, np.mean(self.loss(outputs[-1], y))))
                print(y.flatten()[:10])
                print(outputs[-1].flatten()[:10])
            
            
        
def train():
    batch_size = 1000
    x = np.vstack(np.linspace(-5, 5, batch_size))
    y = x ** 2 + 1 + np.random.random([batch_size, 1])

    model = NN([1, 30, 1])
    model.fit(x, y, num_epochs=100000)

train()

