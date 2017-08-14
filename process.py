import glob
import re
import os
from bs4 import BeautifulSoup, Comment
import nltk
from nltk.corpus import wordnet
import requests
import zipfile
import io

FILE_PATHS = [
    'data/ati/ati_website/html/tipitaka/mn/mn.*.html',
    'data/ati/ati_website/html/tipitaka/dn/dn.*.html',
    'data/ati/ati_website/html/tipitaka/an/**/an*.html',
    'data/ati/ati_website/html/tipitaka/sn/**/sn*.html',
]

META_KEYS = [
    'AUTHOR', 'AUTHOR_SHORTNAME', 'NIKAYA', 'NUMBER', 'MY_TITLE', 'SUBTITLE', 'SUMMARY',
    'TYPE', 'NIKAYA_ABBREV', 'SECTION'
]

ZIPFILE_NAME = 'ati-legacy-2015.12.20.21.zip'
ZIPFILE_URL = 'http://www.accesstoinsight.org/tech/download/bulk/ati-legacy-2015.12.20.21.zip'

OUTPUT_PATH = 'data'

RE_META_BLOCK = re.compile(
    r'Begin ATIDoc metadata dump:(.*)End ATIDoc metadata dump.*', flags=re.DOTALL)

RE_META_ENTRY = re.compile(r'\[([A-Z0-9_]*)\]=?([^\[]*)')

RE_META_VALUE = re.compile(r'\{([^}]*)\}', flags=re.DOTALL)


def convert_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def parse_files():

    sentences = []

    for path in FILE_PATHS:
        for inpath in glob.iglob(path):

            with open(inpath) as infile:

                html = infile.read()
                soup = BeautifulSoup(html, 'html.parser')
                text = extract_text(soup)
                processed = process_text(text)

                print('processed {}'.format(inpath))
                sentences.extend(processed)

    with open('data/sentences.txt', 'w') as outfile:
        outfile.write('\n'.join(sentences))
        print('wrote sentences -> {}'.format(outfile.name))


def process_text(text):

    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentences = []

    # paragraph -> sentences
    for line in text.split('\n'):
        #sentences = nltk.sent_tokenize(line)

        # sentence -> words
        for sentence in nltk.sent_tokenize(line):

            words = nltk.word_tokenize(sentence)
            words = [w.lower() for w in words if len(w) > 0]

            # pos tag + wordnet lemmatize
            tagged = nltk.pos_tag(words)
            tagged = [(t[0], convert_tag(t[1])) for t in tagged]
            words = [lemmatizer.lemmatize(t[0], pos=t[1]) for t in tagged]

            # remove remaining punctuation & extraneous characters
            words = [w for w in words if len(w) > 1 or w in {'a', 'i', 'o'}]
            words = [w for w in words if re.match(r'^[a-z/-]+$', w)]

            # reassemble sentence string
            string = ' '.join(words)
            string = re.sub(r' n\'t ', ' not ', string)  # replace n't -> not

            sentences.append(string)

    return sentences


def extract_text(soup):

    chapter = soup.find('div', {'class': 'chapter'})

    if chapter is None:
        chapter = soup.find('div', {'id': 'COPYRIGHTED_TEXT_CHUNK'})

    paragraphs = []

    for child in chapter.find_all(True, recursive=False):
        text = child.text.strip()

        text = re.sub(r'\[[0-9]+\]', '', text)  # strip footnotes TODO handle
        text = re.sub(r'^[0-9]+\.', '', text)  # strip leading numbers like X.
        text = re.sub(r'\(\s*[0-9]+\s*\)', '', text)  # strip numerical notes like (X)
        text = re.sub(r'\[([^\]]*)\]', '', text)  # strip notes like [X]
        text = re.sub(r'\(([^\)]*)\)', '', text)  # strip notes like (X)

        # convert non-ASCII characters
        text = re.sub(r'Ñ', 'N', text)
        text = re.sub(r'Ā', 'A', text)
        text = re.sub(r'ḷ', 'l', text)
        text = re.sub(r'ā', 'a', text)
        text = re.sub(r'ä', 'a', text)
        text = re.sub(r'ṇ', 'n', text)
        text = re.sub(r'ḍ', 'd', text)
        text = re.sub(r'ñ', 'n', text)
        text = re.sub(r'ṅ', 'n', text)
        text = re.sub(r'ü', 'u', text)
        text = re.sub(r'ū', 'u', text)
        text = re.sub(r'ī', 'i', text)
        text = re.sub(r'ṭ', 't', text)
        text = re.sub(r'ṃ', 'm', text)
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', '\'', text)

        # strip / fix minor artifacts
        text = re.sub(r'&', 'and', text)
        text = re.sub(r'—', '–', text)
        text = re.sub(r'[§¶]', '', text)
        text = re.sub(r'^\s+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\"\s+', '"', text)

        if child.name == 'p':
            paragraphs.append(text)

        elif child.name == 'div' and 'freeverse' in child.attrs.get('class'):
            paragraphs.append(text)

    return '\n'.join(paragraphs)


def parse_meta(soup):

    meta = {}

    # comment blocks -> metadata string.
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):

        contents = RE_META_BLOCK.search(str(comment))
        if contents:

            #string line -> key / value pair.
            for tag in RE_META_ENTRY.findall(contents.group(1)):
                key = tag[0].strip()

                if key in META_KEYS:
                    values = RE_META_VALUE.findall(tag[1])
                    meta[key] = values[
                        0] if values else None  # TODO multiple values possible

    return meta


def maybe_download():

    if not os.path.exists(os.path.join(OUTPUT_PATH, 'ati')):
        print('downloading {}...'.format(ZIPFILE_URL))
        response = requests.get(ZIPFILE_URL)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print('extracting files...')
            z.extractall(os.path.join(OUTPUT_PATH, 'ati'))


if __name__ == '__main__':

    maybe_download()
    parse_files()
