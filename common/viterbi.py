"""
Viterbi encoding and decoding. Implementation found online at
https://github.com/DanielBonanno/Convolutional-Encoders-and-Viterbi-Decoder/tree/master

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025
"""
from operator import xor
import numpy as np
import itertools
import copy
import math

#Global Variables showing the Shift Register States
SHIFT_REGISTERS_K2 = [0,0]
SHIFT_REGISTERS_K5 = [0,0,0,0,0]

#Encoder A, when K=2. Works only for 1 bit.
def bit_encoder_K2(bit_in):
    global SHIFT_REGISTERS_K2
    out_1 = xor(bit_in, SHIFT_REGISTERS_K2[1])
    out_2 = xor(xor(bit_in, SHIFT_REGISTERS_K2[0]), SHIFT_REGISTERS_K2[1])
    SHIFT_REGISTERS_K2 = [bit_in, SHIFT_REGISTERS_K2[0]]
    return [out_1, out_2]

#Encoder B, when K=5. Works only for 1 bit.
def bit_encoder_K5(bit_in):
    global SHIFT_REGISTERS_K5
    out_1 = xor(xor(xor(bit_in, SHIFT_REGISTERS_K5[0]), SHIFT_REGISTERS_K5[2]), SHIFT_REGISTERS_K5[4])
    out_2 = xor(xor(xor(xor(bit_in, SHIFT_REGISTERS_K5[1]), SHIFT_REGISTERS_K5[2]), SHIFT_REGISTERS_K5[3]), SHIFT_REGISTERS_K5[4])
    SHIFT_REGISTERS_K5 = [bit_in, SHIFT_REGISTERS_K5[0], SHIFT_REGISTERS_K5[1], SHIFT_REGISTERS_K5[2], SHIFT_REGISTERS_K5[3]]
    return [out_1, out_2]

#Endocer to encode a whole bit stream
#Input parameters:
#   k decides which encoder to use,
#   input bit stream
def stream_encoder(k, input_stream):
    input_stream_termination = copy.copy(input_stream)
    for i in range(0, k):
        input_stream_termination.append(0)
    encoded_stream = []
    for i in range(0, len(input_stream_termination)):
        if k == 2 :
            encoded_stream.append(bit_encoder_K2(input_stream_termination[i]))
        else:
            encoded_stream.append(bit_encoder_K5(input_stream_termination[i]))

    return encoded_stream

#Function to simulate the channel
#Input parameters:
#   binary input stream,
#   eb/no value in db
#   if hard values should be output (true if hard values, false if soft values)
def Channel(input_stream, eb_no_db, hard_values, rate):
    out_stream = copy.copy(input_stream)
    eb_no = pow(10, eb_no_db / 10)
    sigma = 1/(math.sqrt(2 * eb_no * rate))
    for i in range(0, len(input_stream)):
        for j in range(0, len(input_stream[i])):
            if input_stream[i][j] == 0:
                mu = 1
            else:
                mu = -1
            soft_value = np.random.normal(mu, sigma)

            if(hard_values == True):
                if(soft_value > 0):
                    out_stream[i][j] = 0
                else:
                    out_stream[i][j] = 1
            else:
                out_stream[i][j] = soft_value
    return out_stream


#This class defines the transition  inside a state
#Each transition has associated with it the output and the next state
#These are dependent on the input bit, however, there is no need to keep that information since this can be known
#from the name. (Ex: Transition_Data_0 refers to transition data if the input is 0)
class Transition_Data:
    def __init__(self):
       # self.input_bit = None
        self.output_bits = []
        self.next_state = []

#This class defines every state in the trellis. Every state has data associated with it (state information).
#Furthermore, function to initialise states in the first layer and to assign the transitions from the state are created.
class State:
    def __init__(self):
        self.state_name = None           #name of state as a binary list
        self.previous_state = None       #state from which the current state came from
        self.running_cost = 0            #the running cost at this state
        self.possible_transitions = []   #possible transitions from this state

    #for the first layer, the states should have a large cost, since these shouldn't exist
    def initialisation(self):
        self.running_cost = 100000

    #each state is assigned the 2 transitions that are possible from within it, based on the fact that 1 input bit is entered
    def assign_transitions(self, k):
            transition_0 = Transition_Data()
            transition_1 = Transition_Data()

            #transition_0.input_bit = 0
            if(k == 2):
                transition_0.output_bits = [xor(0, self.state_name[1]),  xor(xor(0, self.state_name[0]), self.state_name[1])]
            else:
                transition_0.output_bits = [xor(xor(xor(0, self.state_name[0]), self.state_name[2]), self.state_name[4]) , xor(xor(xor(xor(0, self.state_name[1]), self.state_name[2]), self.state_name[3]), self.state_name[4])]
            transition_0.next_state = ([0] + self.state_name[:len(self.state_name)-1])
            self.possible_transitions.append(transition_0)

            #transition_1.input_bit = 1
            if(k == 2):
                transition_1.output_bits = [xor(1, self.state_name[1]),  xor(xor(1, self.state_name[0]), self.state_name[1])]
            else:
                transition_1.output_bits = [xor(xor(xor(1, self.state_name[0]), self.state_name[2]), self.state_name[4]) , xor(xor(xor(xor(1, self.state_name[1]), self.state_name[2]), self.state_name[3]), self.state_name[4])]
            transition_1.next_state = ([1] + self.state_name[:len(self.state_name)-1])
            self.possible_transitions.append(transition_1)

#The trellis can be viewed as a series of layers. Each layer has a number of states (2^K, where K is the constraint length)
#Functions to initialise the first layer of the trellis and to create a new layer are also present
class Layer:

    def __init__(self):
        self.number_of_states = 4
        self.list_of_states = []

    #each layer has 2^k possible states. these are created within the layer
    def creation(self, k):
        self.number_of_states = pow(2,k)                             #the number of states in a layer
        bin_nos = list(itertools.product([0, 1], repeat=k))          #all the possible names for the states

        for i in range(0, self.number_of_states):                  #assign the states to the list of states and name each state
            self.list_of_states.append(State())
            self.list_of_states[i].state_name = list(bin_nos[i])
            self.list_of_states[i].assign_transitions(k)


    #for the first layer, call the initialisation method on all the states except the all 0s
    def layer_0_initilisation(self):
        for i in range(0, self.number_of_states):
            if(i!=0):
                self.list_of_states[i].initialisation()

#The following function obtains the cost between 2 binary streams. It returns the square error.
#For hard decoding, the values will be integers, whereas for soft decoding, they will be real numbers
def Get_Cost(stream_a, stream_b):
    return np.sum(pow(np.array(stream_a - np.array(stream_b)),2))

#The following function performs Viterbi decoding based on the decoder used (k selects which decoder)
#hard_decoding is set to true when hard decoding is performed. if soft_decodingis to be perform, it should be set to false
def Decoder(k, input_stream, hard_decoding):
    #get the number of tuples that are in the output (pairs)
    input_stream_tuples = len(input_stream)

    #create the trellis and initlize it
    trellis = [Layer()]
    trellis[0].creation(k)
    trellis[0].layer_0_initilisation()

    #continue creating the trellis layer by layer
    for layer in range(0,input_stream_tuples):
        #layer n-1
        previous_layer = trellis[layer]
        #layer n
        new_layer = Layer()
        new_layer.creation(k)

        for new_state in range(0, new_layer.number_of_states):
            #check the cost for every possible transition between states from layer n-1 to layer n
            current_state = new_layer.list_of_states[new_state]
            temp = 1000000000
            for state in range(0, previous_layer.number_of_states):
                previous_state = previous_layer.list_of_states[state]
                for transition in range(0, len(previous_state.possible_transitions)):
                    transition_data = previous_state.possible_transitions[transition]
                    if(current_state.state_name == transition_data.next_state):
                        compare_to = copy.copy(transition_data.output_bits)
                        #here we must take into consideration the fact that, if soft decoding is used, we must compare
                        #the received value to -1 or 1, since these values represent binary 0 and binary 1 respectively
                        if (hard_decoding == False):
                         for i in range(0, len(compare_to)):
                            if (compare_to[i] == 0):
                                compare_to[i] = 1
                            else:
                                compare_to[i] = -1
                        cost = previous_state.running_cost + Get_Cost(compare_to, input_stream[layer])
                        #keep only the lowest cost transition
                        if(cost<temp):
                            temp = cost
                            current_state.previous_state = previous_state
                    current_state.running_cost = temp
        #append the new layer to the trellis
        trellis.append(new_layer)

    #obtain the output stream by taking the first bit of the states when doing the back pass
    #we can do this since there is only 1 bit input
    #this list is then reversed and the termination bits are dropped
    output_stream = []
    current_state = trellis[input_stream_tuples].list_of_states[0]
    for layer in range(input_stream_tuples,0,-1):
        output_stream.append(current_state.state_name[0])
        current_state = current_state.previous_state
    output_stream = output_stream[::-1]
    output_stream = output_stream[:len(output_stream)-k]
    return output_stream


