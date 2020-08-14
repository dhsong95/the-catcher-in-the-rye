from collections import Counter
from collections import defaultdict

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import process_text_data
from utils import load_epub


def get_word_frequency_information(book):
    book_bow = defaultdict(list)

    stopword = stopwords.words('english')

    for chapter, text in book.items():
        for sentence in text:
            for word in sentence.split():
                if word in stopword:
                    continue
                book_bow[chapter].append(word)

    chapters = list()
    words = list()
    frequencies = list()

    for chapter, bow in book_bow.items():
        counter = Counter(bow)
        for word, frequency in counter.most_common():
            chapters.append(chapter)
            words.append(word)
            frequencies.append(frequency)
    word_frequency_information = pd.DataFrame(
        zip(
            chapters,
            words,
            frequencies
        ),
        columns=['chapter', 'word', 'frequency']
    )
    word_frequency_information.to_csv(
        './output/dataframes/q2-word_frequency_information.csv',
        index=False, encoding='utf-8'
    )

    return word_frequency_information


def draw_plot_for_book_length(book_length_information):
    chapter = book_length_information.chapter
    length_char = book_length_information.length_char
    length_word = book_length_information.length_word
    length_unique_word = book_length_information.length_unique_word

    plt.figure(figsize=(12, 9))
    sns.lineplot(x=chapter, y=length_char, label='length by character')
    sns.lineplot(x=chapter, y=length_word, label='length by word')
    sns.lineplot(
        x=chapter, y=length_unique_word,
        label='length by unique word'
    )

    plt.xlabel('chapter')
    plt.ylabel('length')
    plt.title('How long each chapter is')

    plt.xticks(chapter)
    plt.legend()

    plt.savefig('./output/figures/q1-book_length_distribution.png')


if __name__ == "__main__":
    epub_path = \
        './ebook/J. D. Salinger - The Catcher in the Rye '\
        '(1951, Penguin Books Ltd).epub'
    book = load_epub(epub_path)
    book = process_text_data(book)

    word_frequency_information = get_word_frequency_information(book)
