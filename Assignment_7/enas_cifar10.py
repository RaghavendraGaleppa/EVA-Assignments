import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, MaxPool2D, DepthwiseConv2D,
        Input,Flatten,Activation, concatenate, BatchNormalization, GlobalAvgPool2D)
from tensorflow.keras import backend as K


def modify_concatenate(layers):
    """
        - This function will concatenate different layers with different height and width
        - For example, there are two layers of shape (32,32,40)  and (16,16,20), it is not possible to concatenate
            these layers as they are. They bigger layer(the one with shape (32,32,40)) needs to be modified to 
            shape (16,16,x) to concatenate it with the smaller layer. This modification is done by tf.nn.space_to_depth.
        - This will modify the the layers to match the height and width of the smallest sized layer
    """

    layer_specs = [] # Hold the height and width of each layer along with the layer
    min_height = 999999
    min_width = 999999
    for layer in layers:
        layer_shape = K.int_shape(layer)
        height = layer_shape[-3] 
        width = layer_shape[-2]

        # Check if the height and width of this layer is the smallest out of all layers and 
        # assign it to min_height and min_weight
        min_height = min(min_height, height)
        min_width = min(min_width, width)
        
        layer_specs.append([layer, height, width])

    # Now its time to modify the layers to match the size of the smallest layer
    modified_layers = [] # This list will hold modified layers
    for layer, height, width in layer_specs:
        if(int(height/min_height) > 1):
            # Modify the layer 
            layer = tf.nn.space_to_depth(layer,block_size=int(height/min_height))
            modified_layers.append(layer)
        else:
            # It implies that the layer is the shortest
            modified_layers.append(layer)
    
    # Now concatenate the modified layers
    layers_concatenated = concatenate(modified_layers)
    return layers_concatenated

"""
    Building Block 1
"""
inputs = Input((32,32,3))

# Layer 1
layer_1 = DepthwiseConv2D((5,5),padding='same',depth_multiplier=8)(inputs)
layer_1 = BatchNormalization()(layer_1)
layer_1 = Activation('relu')(layer_1)

# Layer 2
layer_2 = Conv2D(32,(5,5),padding='same',strides=1)(layer_1)
layer_2 = BatchNormalization()(layer_2)
layer_2 =   Activation('relu')(layer_2)

# Layer 3
layer_3 = Conv2D(32,(5,5),padding='same',strides=1)(layer_2)
layer_3 = BatchNormalization()(layer_3)
layer_3 =   Activation('relu')(layer_3)

# Layer 4
layer_4 = DepthwiseConv2D((5,5), padding='same',depth_multiplier=2)(concatenate([layer_1,layer_3]))
layer_4 = BatchNormalization()(layer_4)
layer_4 =   Activation('relu')(layer_4)

# Layer 5
layer_5 = MaxPool2D((2,2))(concatenate([layer_4, layer_1]))

"""
    Building Block 2
"""

# Layer 6
layer_6 = DepthwiseConv2D((3,3), padding='same',depth_multiplier=2)(layer_5)
layer_6 = BatchNormalization()(layer_6)
layer_6 =   Activation('relu')(layer_6)

# Layer 7: Concatenate layer_1, layer_4, layer_6
inputs_layer_7 = modify_concatenate([layer_1,layer_4,layer_6])
layer_7 = Conv2D(32, (5,5),padding='same',strides=1)(inputs_layer_7)
layer_7 = BatchNormalization()(layer_7)
layer_7 =   Activation('relu')(layer_7)

# Layer 8: Concatenate layer_3, layer_4, layer_6, layer_7
inputs_layer_8 = modify_concatenate([layer_3, layer_4, layer_6, layer_7])
layer_8 = DepthwiseConv2D((3,3), padding='same', depth_multiplier=2)(inputs_layer_8)
layer_8 = BatchNormalization()(layer_8)
layer_8 =   Activation('relu')(layer_8)

# Layer 9: 6 layers are being Concatenated to form the input of layer_9
# Concatenate layer_8, layer_7, layer_6, layer_4, layer_3, layer_1
inputs_layer_9 = modify_concatenate([layer_8,layer_6,layer_1, layer_4, layer_7,layer_3]) 
layer_9 = DepthwiseConv2D((5,5), padding='same',depth_multiplier=2)(inputs_layer_9)
layer_9 = BatchNormalization()(layer_9)
layer_9 =   Activation('relu')(layer_9)

# Layer 10: 5 layers to concatenate
# Concatenate layer_9, layer_8, layer_6, layer_4, layer_1
inputs_layer_10 = modify_concatenate([layer_9, layer_8, layer_6, layer_4, layer_1])
layer_10 = MaxPool2D((2,2))(inputs_layer_10)

"""
    Building Block 3
"""

# Layer 11: 2 layers to concatenate
# Concatenate layer_10, layer_7
inputs_layer_11 = modify_concatenate([layer_10,layer_7])
layer_11 = Conv2D(32,(5,5),padding='same',strides=1)(inputs_layer_11)
layer_11 = BatchNormalization()(layer_11)
layer_11 =   Activation('relu')(layer_11)

# Layer 12: 4 layers to concatenate
# Concatenate layer_11, layer_8, layer_2, layer_4
inputs_layer_12 = modify_concatenate([layer_11, layer_8, layer_2, layer_4])
layer_12 = DepthwiseConv2D((5,5), padding='same', depth_multiplier=2)(inputs_layer_12)
layer_12 = BatchNormalization()(layer_12)
layer_12 =   Activation('relu')(layer_12)

# Layer 13: 5 layers to concatenate
# Concatenate layer_12, layer_11, layer_3, layer_2, layer_6
inputs_layer_13 = modify_concatenate([layer_12, layer_11, layer_3, layer_2, layer_6])
layer_13 = Conv2D(32, (3,3), padding='same', strides=1)(inputs_layer_13)
layer_13 = BatchNormalization()(layer_13)
layer_13 =   Activation('relu')(layer_13)

# Layer 14: 7 layers to concatenate
# Concatenate layer_13, layer_12, layer_8, layer_3, layer_6, layer_4,  layer_1
inputs_layer_14 = modify_concatenate([layer_13, layer_12, layer_8, layer_3, layer_6, layer_4, layer_1])
layer_14 = DepthwiseConv2D((5,5), padding='same', depth_multiplier=2)(inputs_layer_14)
layer_14 = BatchNormalization()(layer_14)

# Layer 15: 4 layers to concatenate
# Concatenate layer 14, layer_12, layer_4, layer_8
inputs_layer_15 = modify_concatenate([layer_14, layer_12, layer_4, layer_8])
# Create a transition layer from X input kernels to 10 classes
layer_15 = Conv2D(10,(1,1),padding='same',strides=1)(inputs_layer_15)
layer_15 = BatchNormalization()(layer_15)
layer_15 = Activation('relu')(layer_15)

# Add a global averge pool layer
gap_layer_15 = GlobalAvgPool2D()(layer_15)
layer_15 =   Activation('softmax')(gap_layer_15)

