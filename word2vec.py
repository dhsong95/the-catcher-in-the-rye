from tensorflow import keras
from tensorflow.keras.layers import Activation, Dot, Embedding, Reshape
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, window_size=5, embedding_size=30):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = embedding_size

        self.model = self.build_model(vocab_size, embedding_size)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def build_model(self, vocab_size, embedding_size):
        model_center = keras.models.Sequential()
        model_context = keras.models.Sequential()

        model_center.add(
            Embedding(vocab_size, embedding_size)
        )
        model_context.add(
            Embedding(vocab_size, embedding_size)
        )

        center = keras.Input((1,))
        context = keras.Input((1,))

        center_embedding = model_center(center)
        context_embedding = model_context(context)

        model = keras.models.Sequential()
        model.add(Dot(axes=2))
        model.add(Reshape((1,)))
        model.add(Activation('sigmoid'))

        output = model([center_embedding, context_embedding])

        return keras.Model([center, context], output)

    def train(self, sequences, epochs=10):
        for epoch in range(epochs):
            loss = 0.0
            for sequence in sequences:
                sg = skipgrams(
                    sequence,
                    vocabulary_size=self.vocab_size,
                    window_size=self.window_size
                )

                center = np.array(list(zip(*sg[0]))[0])
                context = np.array(list(zip(*sg[0]))[1])
                labels = np.array(sg[1])

                X = [center, context]
                Y = labels

                loss += self.model.train_on_batch(X, Y)

            print(f'Epoch {epoch}, Loss {loss:.4f}')

    def get_embedding(self):
        return self.model.get_weights()[0]
