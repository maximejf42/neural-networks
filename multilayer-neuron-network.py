'''
A simple multi layer neural network to predict the output of XOR behavior

Overview of steps we will be performing :-
1. Create a neural network with 1 hidden layer (4 neurons) and output layer (Line 114-121)
2. Feed forward from 3 inputs each to 4 neurons in hidden layer and from hidder layer to output layer (Line 48)
3. Calculate the output error at output layer (Line 52, 64)
4. Using derivative figure out of how much weights need to be changed at each layer (Line 56, 65)
5. Adjust weights at each layer (Line 71-72)

Repeat above steps in iteration to get right weights

'''
from numpy import exp, dot, random, array

class Neuron():
    
    # Constructure to initialize class object Neuron
    def __init__(self, total_neurons, inputs_per_neuron):
        
        # Will generate a matrix of dimension-  (inputs_per_neuron * total_neurons() with random values
        weights = random.random((inputs_per_neuron, total_neurons))

        # Get values between -1 and 1 range and with mean 0 (Zero)
        self.weights = 2 * weights - 1

class Neural_Network():
    
    # Constructor to initialize class object Neural_Network
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    
    # The derivative of sigmoid, also known as sigmoid prime, 
    # will give us the rate of change (or "slope") of the activation function at the output sum
    def sigmoid_prime(self, output):
        return output * (1 - output)

    # Sigmoid activation function
    # The purpose of the activation function is to transform the input signal 
    # into an output signal and are necessary for neural networks to model complex non-linear patterns that simpler models might miss.
    def sigmoid_activation(self, x):
        return 1 / (1 + exp(-x))

    # Train
    def train(self, data_inputs, data_outputs, iteration):
        for number in xrange(iteration):
            
            l1_output, l2_output = self.perform(data_inputs)

            # At Layer 2, calculate margin of error of the output layer by taking the difference of the predicted output (l2_output)
            # and the actual output (data_outputs)
            l2_error = data_outputs - l2_output

            # Apply the derivative of our sigmoid activation function to the output layer error. 
            # We call this result the delta output sum.
            l2_delta = l2_error * self.sigmoid_prime(l2_output)

            # We have the proposed change in the output layer sum 'l2_delta'
            # We will be using this in the derivative of the output sum function to 
            # determine the new change in weights. Weight adjustment at layer 2 are..
            l2_new_weights = l1_output.T.dot(l2_delta)
            
            # At Layer 1, Calculate error and delta as we did in layer 2
            l1_error = l2_delta.dot(self.layer2.weights.T)
            l1_delta = l1_error * self.sigmoid_prime(l1_output)

            # Adjustmentin the weights at layer 1 will be..
            l1_new_weights = data_inputs.T.dot(l1_delta)
            
            # Apply weight changes to layers 
            self.layer2.weights += l2_new_weights
            self.layer1.weights += l1_new_weights

    
    # Peform the dot product of input and weight matrics and give as an input
    # to sigmoid activation function 
    def perform(self, inputs):
        
        # Pass 1 - Input layer to layer 1 (hidden layer)
        l1_output = self.sigmoid_activation(dot(inputs, self.layer1.weights))
        
        # Pass 2 - layer 1 (hidden layer) to layer 2 (output layer)
        l2_output = self.sigmoid_activation(dot(l1_output, self.layer2.weights))
        
        return l1_output, l2_output

    # Predict the output for network
    def predict(self, input):
        hidden, prediction =  self.perform(input)
        print "Predicted output for input value is : {} ".format(prediction)

    # Print weights attached to the layers
    def weights(self):
        print "Layer 1 - Data Input to hidden layer"
        print self.layer1.weights
        
        print "Layer 2 - Hidden layer to output layer"
        print self.layer2.weights


# File is being executed a main program
if __name__ == "__main__":
    
    #Seed the random number generator,so it generates the same numbers
    # every time the program runs.
    random.seed(1)
    
    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    data_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    data_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
    total_iteration = 60000

    # Hidden layer -  layer 1 ( 4 neurons, 3 inputs from data input)
    layer1 = Neuron(4,3)

    # Output layer - layer 2 (1 neuron, 4 inputs from layer 1)
    layer2 = Neuron(1,4)

    # Main class object for neural network with 2 Neuron
    network = Neural_Network(layer1, layer2)
    
    # Shows the initial random weights with which our 
    # neural network will start training 
    print network.weights()

    # Train our neural network
    network.train(data_inputs, data_outputs, total_iteration)

    # Adjusted weights after training network
    print network.weights()

    # Network is trained, lets predict the output
    print network.predict([1, 1, 0])


    






