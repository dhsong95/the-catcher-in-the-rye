from collections import Counter
from collections import defaultdict
import re

from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd


def _nltk_settings(data_path):
    nltk.data.path.append(data_path)


def _save_apostrophe_dataframe(book):
    text = [t.lower() for t in book.values()]
    text = ' '.join(text)
    pattern = r'([\w]*\'[\w]*)'
    apostrophes = re.findall(pattern, text)
    counter = Counter(apostrophes)
    apostrophe_df = pd.DataFrame(
        counter.most_common(),
        columns=['apostrophe', 'frequency']
    )

    apostrophe_df.to_csv(
        './output/dataframes/q2-apostrophe_dict.csv',
        index=False
    )
    return apostrophe_df


def _get_apostrophe_dict(filename):
    with open(filename, 'r') as f:
        apostrophe_dict = defaultdict(str)
        for line in f.readlines():
            src, dst = line.split('\t')
            apostrophe_dict[src] = dst
    return apostrophe_dict


def process_apostrophe(text, filename='./apostrophe_dict.txt'):
    apostrophe_dict = _get_apostrophe_dict(filename)
    for src, dst in apostrophe_dict.items():
        text = text.replace(src, dst)
    return text


def _get_wordnet_pos(pos):
    if pos.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif pos.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif pos.startswith('R'):
        return nltk.corpus.wordnet.ADV

    return nltk.corpus.wordnet.NOUN


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = list()
    for word, pos in pos_tag(wordpunct_tokenize(text)):
        wordnet_pos = _get_wordnet_pos(pos)
        word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        if re.match(r'[\w]+', word):
            words.append(word)
    return ' '.join(words)


def process_text_data(book):
    _nltk_settings(data_path='./nltk_data/')
    _save_apostrophe_dataframe(book)

    book_processed = defaultdict(list)

    for chapter in book.keys():
        text = book[chapter]
        text = text.lower()

        text = process_apostrophe(text)
        tokenizer = PunktSentenceTokenizer()
        for sentence in tokenizer.tokenize(text):
            sentence = lemmatize_text(sentence)
            if len(sentence):
                book_processed[chapter].append(sentence)
    return book_processed
