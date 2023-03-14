
import tensorflow as tf

class TrainerConfig:
    # Optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    # Checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, model_config, train_dataset, train_dataset_len, test_dataset, test_dataset_len, config):
        self.train_dataset = train_dataset.batch(config.batch_size)
        self.train_dataset_len = train_dataset_len
        self.test_dataset = test_dataset
        self.test_dataset_len = None
        self.test_dist_dataset = None
        if self.test_dataset:
            self.test_dataset = test_dataset.batch(config.batch_size)
            self.test_dataset_len = test_dataset_len
        self.config = config
        self.tokens = 0
        self.strategy = tf.distirbute.OneDeviceStrategy("GPU:0")
        if len(tf.config.list_physical_devices("GPU")) > 1:
            self.strategy = tf.distribute.MirroredStartegy()

        with self.strategy.scope():
            self.model = model(**model_config)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            if self.test_dataset:
                self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

    def save_checkpoints(self):
        if self.config.ckpt_path is not None:
            self.model.save_weights(self.config.ckpt_path)

    def train(self):
        pass
