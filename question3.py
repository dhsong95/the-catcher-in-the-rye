from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim
import pandas as pd

from preprocess import process_text_data
from utils import load_epub
from word2vec import Word2Vec


def remove_stopword(book):
    corpus = list()
    stopword = stopwords.words('english')
    for _, sentences in book.items():
        for sentence in sentences:
            stopword_removed = list()
            for word in sentence.split():
                if word in stopword:
                    continue
                stopword_removed.append(word)
            if len(stopword_removed) > 1:
                corpus.append(' '.join(stopword_removed))
    return corpus


def tokenize_corpus(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    sequences = tokenizer.texts_to_sequences(corpus)

    return tokenizer, sequences


if __name__ == '__main__':
    epub_path = \
        './ebook/J. D. Salinger - The Catcher in the Rye '\
        '(1951, Penguin Books Ltd).epub'
    book = load_epub(epub_path)
    book = process_text_data(book)
    book[0] = [' '.join(book[0])]

    corpus = remove_stopword(book)
    tokenizer, sequences = tokenize_corpus(corpus)

    word2idx = tokenizer.word_index

    vocab_size = len(word2idx) + 1
    window_size = 10
    embedding_size = 20
    word2vec = Word2Vec(vocab_size, window_size, embedding_size)

    word2vec.train(sequences, epochs=15)
    embedding = word2vec.get_embedding()
    with open('./output/embedding/word_embedding.txt', 'w') as f:
        f.write(f'{vocab_size-1} {embedding_size}\n')
        for word, idx in word2idx.items():
            word_vec = ' '.join(map(str, list(embedding[idx, :])))
            f.write(f'{word} {word_vec}\n')

    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        fname='./output/embedding/word_embedding.txt'
    )

    targets = [
        'holden', 'phoebe', 'ackley', 'stradlater', 'jane', 'janine', 'rye'
    ]

    for target in targets:
        print(f'What Words are most Similarit to {target}')

        similar_words = w2v.most_similar(positive=[target])
        similar_df = pd.DataFrame(
            similar_words, columns=['word', 'similarity']
        )
        similar_df.to_csv(
            f'./output/dataframes/q3-similar_with_{target}.csv', index=False
        )
