from translator.translator import EATranslator
from util import create_dataset

PATH = '../Resource/por.txt'
DIC_PATH = '../Resource/en-pt.csv'


def clean_text(text_list):
    stop_list = ['Article', 'Adverb', 'Noun', 'Adjective', 'Verb', '[', ']', '\n', '\\']
    for i, text in enumerate(text_list):

        for word in stop_list:
            text = text.replace(word, '')

        text_list[i] = text
    return text_list


def main():
    en, pt = create_dataset(PATH, 30000, 0)


if __name__ == '__main__':
    main()
