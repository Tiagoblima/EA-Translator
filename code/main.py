from util import create_dataset, get_vocabulary
from translator.translator import EATranslator, get_dic
import os, re

PATH = '../Resource/por.txt'
DIC_PATH = '../Resource/PORTUGUE.TXT'


def clean_text(text_list):
    stop_list = ['Article', 'Adverb', 'Noun', 'Adjective', 'Verb', '[', ']', '\n', '\\']
    for i, text in enumerate(text_list):

        for word in stop_list:
            text = text.replace(word, '')

        text_list[i] = text
    return text_list


def main():
    """gu, pt = create_dataset(PATH, 1000)
    print(pt[1])
    eatranslator = EATranslator(pt, pt[1])
    eatranslator.run_evaluations()"""

    pt_en = get_dic()
    en, pt = create_dataset(PATH, 30000)

    eatranslator = EATranslator(pt, en[80], pt[80])
    eatranslator.run_evaluations()
    print(len(pt_en.keys()))


if __name__ == '__main__':
    main()
