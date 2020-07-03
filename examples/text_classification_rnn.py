#!/usr/bin/env python
#coding: utf-8

#Text classification with an RNN
#This text classification tutorial trains a [recurrent neural network](https://developers.google.com/machine-learning/glossary/#recurrent_neural_network) on the [IMDB large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) for sentiment analysis.

import importlib
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import os

def data_loader(hyperparams, local_data_dir):
    dataset, info = tfds.load('imdb_reviews/subwords8k', 
                              data_dir=local_data_dir,
                              with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder
    train_dataset = train_dataset.shuffle(hyperparams['BUFFER_SIZE'])
    train_dataset = train_dataset.padded_batch(hyperparams['BATCH_SIZE'], padded_shapes=None)
    test_dataset = test_dataset.padded_batch(hyperparams['BATCH_SIZE'], padded_shapes=None)
    return train_dataset, test_dataset, encoder

def define_model(encoder):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

class MovieReviewClassification(object):
    def __init__(self, learning_rate=1e-4, batch_size=64, epochs=2, local_data_dir='/app/tensorflow_datasets'):
        hyperparams = {'BUFFER_SIZE': 10000, 'BATCH_SIZE': batch_size}
        self.model_file = "lstm_trained"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_dataset, self.test_dataset, self.encoder = data_loader(hyperparams, local_data_dir)
        
    def train(self):
        model = define_model(self.encoder)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      metrics=['accuracy'])
        history = model.fit(self.train_dataset, epochs=self.epochs,
                            validation_data=self.test_dataset,
                            validation_steps=30)
        model.save(self.model_file)
        test_loss, test_acc = model.evaluate(self.test_dataset)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", default="1e-4", help="Learning rate for the Keras optimizer")
    parser.add_argument("-bsz", "--batch_size", default="64", help="Batch size for each step of learning")
    parser.add_argument("-e", "--epochs", default="2", help="Number of epochs in each trial")
    args = parser.parse_args()
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    model = MovieReviewClassification(learning_rate, batch_size, epochs, local_data_dir="/app/tensorflow_datasets")
    model.train()