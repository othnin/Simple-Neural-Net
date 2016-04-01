import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
#import sklearn
#import sklearn.linear_model
#import matplotlib
#from sklearn.datasets.tests.test_svmlight_format import datafile

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = training_set_inputs [:, 0].min() - .5, training_set_inputs [:, 0].max() + .5
    y_min, y_max = training_set_inputs [:, 1].min() - .5, training_set_inputs [:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(training_set_inputs [:, 0], training_set_inputs [:, 1], c=training_set_outputs, cmap=plt.cm.Spectral)
    

    
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def predict(self, model, training_set_inputs):
        output_from_layer1 = self.__sigmoid(np.dot(training_set_inputs, self.layer1.synaptic_weights))
        #output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer2 = output_from_layer1.dot(self.layer2.synaptic_weights)
        exp_scores = np.exp(output_from_layer2)
        probs = exp_scores / np.sum(exp_scores,  keepdims=True)
        return np.argmax(probs, axis=1)
    

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
        #output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer2 = output_from_layer1.dot(self.layer2.synaptic_weights)
        exp_scores = np.exp(output_from_layer2)
        probs = exp_scores / np.sum(exp_scores, keepdims=True)
        return output_from_layer1, probs#np.argmax(probs, axis=1)#output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print "-----Layer 1 (3 neurons, each with 2 inputs)----"
        print self.layer1.synaptic_weights
        print "----Layer 2 (2 neuron, with 3 inputs)----"
        print self.layer2.synaptic_weights

if __name__ == "__main__":
    training_set_outputs = []
    training_set_inputs , raw_y = sklearn.datasets.make_moons(2000, noise=0.20)
    for item in raw_y:
        training_set_outputs.append([item])
        
    #Seed the random number generator
    np.random.seed(1)

    # Create layer 1 (3 neurons, each with 2 inputs)
    layer1 = NeuronLayer(3, 2)
    # Create layer 2 (2 neuron with 3 inputs)
    layer2 = NeuronLayer(2, 3)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)
    plot_decision_boundary(lambda x: neural_network.predict(training_set_inputs, x))
    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new input.
    print "Stage 3) Considering a new input [2, -1] -> ?: "
    hidden_state, output = neural_network.think(np.array([2, -1]))
    print output