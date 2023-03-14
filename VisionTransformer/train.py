
from tensorflow.keras import datasets
from trainUtils import TrainerConfig


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Prepare the dataset
train_images = tf.cast(train_images.reshape(-1, 3, 32, 32), dtype="float32")
test_images = tf.cast(test_images.reshape(-1, 3, 32, 32), dtype="float32")
train_images, test_images = train_images / 255.0, test_images / 255.0

train_x = tf.data.Dataset.from_tensor_slices(train_images,)
train_y = tf.data.Dataset.from_tensor_slices(train_labels,)
train_dataset = tf.data.Dataset.zip((train_x, train_y))
test_x = tf.data.Dataset.from_tensor_slices(test_images,)
test_y = tf.data.Dataset.from_tensor_slices(test_labels,)
test_dataset = tf.data.Dataset.zip((test_x, test_y))

# Training configuration
train_config = TrainerConfig(
    max_epochs=10,
    batch_size=64,
    learning_rate=1e-3
)

# Model configuration
model_config = {
    "image_size":32,
    "patch_size":4,
    "num_classes":10,
    "dim":64,
    "depth":3,
    "heads":4,
    "mlp_dim":128
}