
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K

from CustomLoss import constrastive_loss_with_margin


def initialize_base_network():
    input = Input(shape=()) # Put the shape here
    x = Flatten()(input)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    return Model(inputs=input, outputs=x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon))

def eucl_dist_output_shape(shapes):
    shape1,  shape2 = shapes
    return (shape1[0], 1)


base_network = initialize_base_network()

input_a = Input(shape=()) # Put the shape here
input_b = Input(shape=())

vect_output_a = base_network(input_a)
vect_output_b = base_network(input_b)

output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
model = Model([input_a, input_b], output)

rms = RMSprop()
model.compile(loss=constrastive_loss_with_margin(margin=1), optimizer=rms)
