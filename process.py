import glob
import re
import os
from bs4 import BeautifulSoup, Comment
import nltk
from nltk.corpus import wordnet

FILE_PATHS = [
    'data/source/ati/html/tipitaka/mn/mn.*.html',
    'data/source/ati/html/tipitaka/dn/dn.*.html',
    'data/source/ati/html/tipitaka/an/**/an*.html',
    'data/source/ati/html/tipitaka/sn/**/sn*.html',
]

META_KEYS = [
    'AUTHOR', 'AUTHOR_SHORTNAME', 'NIKAYA', 'NUMBER', 'MY_TITLE', 'SUBTITLE', 'SUMMARY', 'TYPE', 'NIKAYA_ABBREV',
    'SECTION'
]

OUTPUT_PATH = 'data'

RE_META_BLOCK = re.compile(r'Begin ATIDoc metadata dump:(.*)End ATIDoc metadata dump.*', flags=re.DOTALL)

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


def parse_file(inpath):

    text_path = os.path.join(OUTPUT_PATH, 'text')
    processed_path = os.path.join(OUTPUT_PATH, 'processed')

    for p in [text_path, processed_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    with open(inpath) as infile:

        html = infile.read()
        soup = BeautifulSoup(html, 'html.parser')

        # raw text & metadata
        meta = parse_meta(soup)
        text = parse_text(soup)

        # get filename metadata
        book = meta['NIKAYA_ABBREV'].lower()
        number = meta['NUMBER']
        section = meta['SECTION'] or 1

        # TODO save metadata

        # save raw text
        outfile_name = '{}-{}-{}.txt'.format(book, section, number)
        outpath = os.path.join(text_path, outfile_name)

        with open(outpath, 'w') as outfile:
            outfile.write(text)
            print('{} -> {}'.format(inpath, outpath))

        # preprocess
        processed = preprocess(text)

        # save preprocessed text
        outfile_name = '{}-{}-{}.txt'.format(book, section, number)
        outpath = os.path.join(processed_path, outfile_name)

        with open(outpath, 'w') as outfile:
            outfile.write(processed)
            print('{} -> {}'.format(inpath, outpath))


def preprocess(text):

    lines = text.split('\n')
    processed = []
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # paragraph -> sentences
    for line in lines:
        sentences = nltk.sent_tokenize(line)
        out = []

        # sentence -> words
        for sentence in sentences:

            words = nltk.word_tokenize(sentence)
            words = [w.lower() for w in words]

            # split hyphenated words
            split = [w.split('-') for w in words]
            words = [w for s in split for w in s]
            words = [w for w in words if len(w) > 0]

            # pos tag + wordnet lemmatize
            tagged = nltk.pos_tag(words)
            tagged = [(t[0], convert_tag(t[1])) for t in tagged]
            words = [lemmatizer.lemmatize(t[0], pos=t[1]) for t in tagged]
            words = [w for w in words if re.match(r'^[a-z/-]+$', w)]

            # reassemble sentence string
            string = ' '.join(words)

            # fix tokenizer quote mangling
            string = re.sub(r'``', '"', string)
            string = re.sub(r'\'\'', '"', string)

            string = re.sub(r' n\'t ', ' not ', string)     # n't -> not
            string = re.sub(r" '([^'])", r" ' \1", string)  # fix quote spacing
            string = re.sub(r'\n', '', string)              # strip any remaining newlines

            out.append(string)

        processed.append('\n'.join(out))

    return '\n'.join(processed)


def parse_text(soup):

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
                    meta[key] = values[0] if values else None  # TODO multiple values possible

    return meta


if __name__ == '__main__':
    for path in FILE_PATHS:
        for filename in glob.iglob(path):
            parse_file(filename)
