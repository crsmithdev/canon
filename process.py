import glob
import re
from bs4 import BeautifulSoup, Comment

FILE_PATHS = [
    'data/source/ati/html/tipitaka/mn/mn.*.html',
    'data/source/ati/html/tipitaka/dn/dn.*.html',
    'data/source/ati/html/tipitaka/an/**/an*.html',
    'data/source/ati/html/tipitaka/sn/**/sn*.html',
    #'data/source/html/tipitaka/mn/mn.007.nypo.html',
]

def parse_page(html):

    soup = BeautifulSoup(html, 'html.parser')

    meta = parse_meta(soup)
    print(meta)
    text = parse_text(soup)

    return meta, text

def parse_text(soup):

    chapter = soup.find("div", {"class": "chapter"})
    if chapter is None:
        chapter = soup.find('div', {'id': 'COPYRIGHTED_TEXT_CHUNK'})
    paragraphs = []

    for child in chapter.find_all(True, recursive=False):
        if child.name == "p":
            content = child.text.strip()
            content = re.sub(r"^[0-9]+\.", "", content)
            content = re.sub(r"\[[0-9]+\]", "", content) # TODO handle notes
            paragraphs.append(content)
        elif child.name == "div" and "freeverse" in child.attrs.get("class"):
            content = child.text.strip()
            content = re.sub(r"\[[0-9]+\]", "", content) # TODO handle notes
            paragraphs.append(content)

    return "\n".join(paragraphs)

def parse_meta(soup):

    kv = {}
    targets = [
        'AUTHOR', 'AUTHOR_SHORTNAME', 'NIKAYA', 'NUMBER', 'MY_TITLE',
        'SUBTITLE', 'SUMMARY', 'TYPE', 'NIKAYA_ABBREV', 'SECTION'
    ]

    for comments in soup.findAll(text=lambda text: isinstance(text, Comment)):
        meta = re.search(
            r" Begin ATIDoc metadata dump:(.*)End ATIDoc metadata dump.*",
            str(comments),
            flags=re.DOTALL)
        if meta:
            tags = re.findall(r"\[([A-Z0-9_]*)\]=?([^\[]*)", meta.group(1))
            for tag in tags:
                key = tag[0].strip()
                values = re.findall(r"\{([^}]*)\}", tag[1], re.DOTALL)
                if key in targets:
                    value = values[0] if len(values) > 0 else None
                    kv[key] = value

    return kv


for path in FILE_PATHS:
    for filename in glob.iglob(path):
        print(filename)
        with open(filename) as file:
            meta, text = parse_page(file)
            #print(meta)
            #print(text)
            print(meta['NIKAYA_ABBREV'], meta['NUMBER'])
            nikaya = meta['NIKAYA_ABBREV']
            number = meta['NUMBER']
            section = meta['SECTION'] or 1
            outfilename = "data/processed/{0}-{1}-{2}.txt".format(nikaya, section, number)
            with open(outfilename, 'w') as outfile:
                outfile.write(text)
