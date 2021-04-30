import tensorflow as tf 
import tensorflow_datasets as tfds 
from model import ResNet

def  preprocess(features):
    return tf.cast(features["image"], tf.float32) / 255., features["label"]


resnet = ResNet(10)
resnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
dataset = tfds.load("mnist", split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)
resnet.fit(dataset, epochs=1)