# https://builtin.com/machine-learning/nn-models
#
#

# There are a total of three layers in the model. The first layer is the input
# layer and has 30 neurons for each of the 30 inputs. The second layer is the
# hidden layer, and it contains 14 neurons by default. The third layer is the
# output layer, and since we have two classes, 0 and 1, we require only one
# neuron in the output layer. The default learning rate is set as 0.001, and
# the number of iterations or epochs is 100.
#
# Remember, there is a huge difference between the terms epoch and iterations.
# Consider a dataset of 2,000 data points. We are dividing the data into batches
# of 500 data points and then training the model on each batch. The number
# of batches to be trained for the entire data set to be trained once is called
# iterations. Here, the number of iterations is four. The number of times the
# entire data set undergoes a cycle of forward propagation and backpropagation
# is called epochs. Since the data above is not divided into batches, iteration
# and epochs will be the same.
#
# The weights are initialized randomly. The bias weight is not added with the main
# input weights, it is maintained separately.
#
# Then the sigmoid activation function and cost function of the neural network are
# defined. The cost function takes in the predicted output and the actual output as
# input, and calculates the cost.
#
# The forward propagation function is then defined. The activations of the input
# layer is calculated and passed on as input to the output layer. All the parameters
# are stored in a dictionary with suitable labels.
#
# The backpropagation function is defined. It takes in the predicted output to perform
# backpropagation using the stored parameter values. The gradients are calculated as
# we discussed above and the weights are updated in the end.
#
# The fit function takes in the input x and desired output y. It calls the weight
# initialization, forward propagation and backpropagation function in that order and
# trains the model. The predict function is used to predict the output of the test set.
# The accuracy function can be used to test the performance of the model. There is also
# a function available for plotting the cost function vs epochs.

class NeuralNet():
    def __init__(self, layers = [30, 14, 1], learning_rate = 0.001, iterations = 100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.Y = None
    def init_weights(self):
        np.random.seed(1)
        self.params['theta_1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['theta_2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)
    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))
    def cost_fn(self, y, h):
        m = len(y)
        cost = (-1/m) * (np.sum(np.multiply(np.log(h), y) + np.multiply((1-y), np.log(1-h))))
        return cost
    def forward_prop(self):
        Z1 = self.X.dot(self.params['theta_1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['theta_2']) + self.params['b2']
        h = self.sigmoid(Z2)
        cost = self.cost_fn(self.Y, h)
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        return h, cost
    def back_propagation(self, h):
        diff_J_wrt_h = -(np.divide(self.Y, h) - np.divide((1 - self.Y), (1 - h)))
        diff_h_wrt_Z2 = h * (1 - h)
        diff_J_wrt_Z2 = diff_J_wrt_h * diff_h_wrt_Z2
        diff_J_wrt_A1 = diff_J_wrt_Z2.dot(self.params['theta_2'].T)
        diff_J_wrt_theta_2 = self.params['A1'].T.dot(diff_J_wrt_Z2)
        diff_J_wrt_b2 = np.sum(diff_J_wrt_Z2, axis = 0)
        diff_J_wrt_Z1 = diff_J_wrt_A1 * (self.params['A1'] * ((1-self.params['A1'])))
        diff_J_wrt_theta_1 = self.X.T.dot(diff_J_wrt_Z1)
        diff_J_wrt_b1 = np.sum(diff_J_wrt_Z1, axis = 0)
        self.params['theta_1'] = self.params['theta_1'] - self.learning_rate * diff_J_wrt_theta_1
        self.params['theta_2'] = self.params['theta_2'] - self.learning_rate * diff_J_wrt_theta_2
        self.params['b1'] = self.params['b1'] - self.learning_rate * diff_J_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * diff_J_wrt_b2
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.init_weights()
        for i in range(self.iterations):
            h, cost = self.forward_prop()
            self.back_propagation(h)
            self.cost.append(cost)
    def predict(self, X):
        Z1 = X.dot(self.params['theta_1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['theta_2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)
    def acc(self, y, h):
        acc = (sum(y == h) / len(y) * 100)
        return acc
    def plot_cost(self):
        fig = plt.figure(figsize = (10,10))
        plt.plot(self.cost)
        plt.xlabel('No. of iterations')
        plt.ylabel('Logistic Cost')
        plt.show()