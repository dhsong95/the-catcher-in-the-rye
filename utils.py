from collections import defaultdict
import re

from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib


def transform_epub_to_html(epub_path):
    book = epub.read_epub(epub_path)
    chapters = defaultdict(str)
    for item in book.get_items():
        name = item.get_name()
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters[name] = item.get_content()
    return chapters


def transform_html_to_text(html):
    blacklist = [
        '[document]', 'noscript', 'header', 'html',
        'meta', 'head', 'input', 'script'
    ]
    soup = BeautifulSoup(html, 'html.parser')
    output = ''
    for text in soup.find_all(text=True):
        if text.parent.name in blacklist:
            continue
        output += (' ' + text)
    return output


def load_epub(epub_path):
    chapters = transform_epub_to_html(epub_path)
    book = defaultdict(str)
    for chapter, html in chapters.items():
        chapter_found = re.search(
            r'The_Catcher_in_the_Rye_split_([\d]{3}).html', chapter
        )
        if chapter_found:
            chapter = int(chapter_found.group(1))
            text = transform_html_to_text(html)
            text = re.sub(r'[\s][\s]+', ' ', text)
            text = text.strip()
            book[chapter] = text
    return book
