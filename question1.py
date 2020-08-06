import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import load_epub


def get_book_length_information(book):
    N = len(book)
    chapters = sorted(list(book.keys()))
    length_by_character = [0] * N
    length_by_word = [0] * N
    length_by_unique_word = [0] * N

    for chapter in chapters:
        length_by_character[chapter] = len(book[chapter])
        length_by_word[chapter] = len(book[chapter].split())
        length_by_unique_word[chapter] = len(set(book[chapter].split()))

    book_length_information = pd.DataFrame(
        zip(
            chapters,
            length_by_character,
            length_by_word, length_by_unique_word
        ),
        columns=['chapter', 'length_char', 'length_word', 'length_unique_word']
    )
    book_length_information.to_csv(
        './output/dataframes/q1-book_length_information.csv',
        index=False, encoding='utf-8'
    )

    return book_length_information


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
    book_length_information = get_book_length_information(book)
    draw_plot_for_book_length(book_length_information)
