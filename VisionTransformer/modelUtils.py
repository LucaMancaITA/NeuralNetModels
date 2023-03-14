
# Import modules
import math
import six
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K



def gelu(x):
    """Gaussian Error Linear Unit.
    Smoother version of the ReLU.

    Args:
        x (float): float tensor to perform activation

    Returns:
        float: "x" with the GeLU activation applied
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi)  * (x + 0.044715 * tf.pow(x, 3)))
    ))
    return x * cdf


def get_activation(identifier):
    """Maps an identifier to a Python function, e.g., "relu" -> "tf.nn.relu".

    Args:
        identifier ([type]): [description]
    """

    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


def model_sanity_check(model, image_size, n_classes):
    """Checks if the NN model is returning the expected output.

    Args:
        model (tf.keras.Model): tensorflow model
        image_size (int): image input size
        n_classes (int): number of output classes
    """
    x = tf.random.uniform(shape=(1, 3, image_size, image_size))
    y = model(x)
    assert (y.shape == (1, n_classes))
    print("_________________________________________")
    print("Model sanity check:")
    print("\nInput image:   ", x.shape)
    print("Output image:  ", y.shape)
    print()


def get_model_memory_usage(batch_size, model, model_config):

    print("_________________________________________")
    print("Computing memory usage:")
    features_mem = 0 # Initialize memory for features.
    float_bytes = 4.0 # Multiplication factor as all values we store would be float32.
    img_size = model_config["image_size"]
    x = tf.random.uniform((1, 3, img_size, img_size))

    for layer in model.layers:
        x = layer.call(x)
        out_shape = x.shape
        #out_shape = layer.output_shape

        if type(out_shape) is list:   #e.g. input layer which is a list
            out_shape = out_shape[0]
        else:
            if len(out_shape) == 4:
                out_shape = [out_shape[1], out_shape[2], out_shape[3]]
            elif len(out_shape) == 3:
                out_shape = [out_shape[1], out_shape[2]]

        # Multiply all shapes to get the total number per layer.
        single_layer_mem = 1
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s

        single_layer_mem_float = single_layer_mem * float_bytes # Multiply by 4 bytes (float)
        single_layer_mem_MB = single_layer_mem_float/(1024**2)  # Convert to MB

        print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB)
        features_mem += single_layer_mem_MB  # Add to total feature memory count

    # Calculate Parameter memory
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes)/(1024**2)
    print("_________________________________________")
    print("Memory for features in MB is:", features_mem*batch_size)
    print("Memory for parameters in MB is: %.2f" %parameter_mem_MB)

    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB  #Same number of parameters. independent of batch size

    total_memory_GB = total_memory_MB/1024

    print("_________________________________________")
    print("Minimum memory required to work with this model is: %.4f" %total_memory_GB, "GB")

    return total_memory_GB