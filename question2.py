from collections import Counter
from collections import defaultdict

from nltk.corpus import stopwords
from wordcloud import WordCloud
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


def get_top_word_frequency_information(word_freq_info, topn=100):
    words = word_freq_info.word.unique()
    frequencies =\
        word_freq_info.groupby(by='word').sum().loc[words, 'frequency'].values

    top_word_freq_info = pd.DataFrame(
        zip(
            words,
            frequencies
        ),
        columns=['word', 'frequency']
    )

    top_word_freq_info = top_word_freq_info.sort_values(
        by='frequency', ascending=False, ignore_index=True
    )
    top_word_freq_info = top_word_freq_info.head(topn)

    top_word_freq_info.to_csv(
        './output/dataframes/q2-top_word_frequency_information.csv',
        index=False, encoding='utf-8'
    )
    return top_word_freq_info


def draw_barplot_top_word_frequency_information(top_word_freq_info, badwords):
    badword_index =\
        top_word_freq_info[top_word_freq_info.word.isin(badwords)].index
    non_badword_index =\
        top_word_freq_info[~top_word_freq_info.word.isin(badwords)].index

    plt.figure(figsize=(12, 9))
    plt.bar(
        x=badword_index,
        height=top_word_freq_info.loc[badword_index, 'frequency'],
        color='r', label='badwords{goddam, hell, damn, bastard}'
    )
    plt.bar(
        x=non_badword_index,
        height=top_word_freq_info.loc[non_badword_index, 'frequency'],
        color='b'
    )

    plt.legend()
    plt.xticks([])
    plt.xlabel('word')
    plt.ylabel('frequency')
    plt.title('Top Word Frequency Distribution. (feat. badwords)')
    plt.savefig('./output/figures/q2-top_word_frequency.png')


def draw_wordcloud_top_word_frequency_information(top_word_freq_info):
    word_freq = {word: freq for word, freq in zip(
        top_word_freq_info['word'], top_word_freq_info['frequency']
    )}
    wc = WordCloud(
        background_color='white',
        max_words=1000,
        contour_width=3,
        contour_color='firebrick',
        random_state=2020
    )
    wc = wc.generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 12))
    plt.imshow(wc)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title('WordCloud for Top Words in Novel')
    plt.savefig('./output/figures/q2-top_word_frequency_wordcloud.png')


if __name__ == "__main__":
    epub_path = \
        './ebook/J. D. Salinger - The Catcher in the Rye '\
        '(1951, Penguin Books Ltd).epub'
    book = load_epub(epub_path)
    book = process_text_data(book)

    word_freq_info = get_word_frequency_information(book)
    top_word_freq_info = get_top_word_frequency_information(
        word_freq_info, topn=200
    )

    badwords = ['goddam', 'hell', 'damn', 'bastard']
    draw_barplot_top_word_frequency_information(top_word_freq_info, badwords)
    draw_wordcloud_top_word_frequency_information(top_word_freq_info)
