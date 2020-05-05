import googletrans
import pandas as pd
from translate import Translator

from translator.translator import EATranslator
from util import create_dataset, TextBlob, get_vocabulary, NotTranslated

PATH = '../Resource/por.txt'
DIC_PATH = '../Resource/en-pt.csv'


def clean_text(text_list):
    stop_list = ['Article', 'Adverb', 'Noun', 'Adjective', 'Verb', '[', ']', '\n', '\\']
    for i, text in enumerate(text_list):

        for word in stop_list:
            text = text.replace(word, '')

        text_list[i] = text
    return text_list


def create_dictionary():
    en, pt = create_dataset(PATH, 10000, 0)
    vocabulary = list(get_vocabulary(en))

    dictionary = {}
    for word in vocabulary:
        print(word)

        try:
            translation = Translator(to_lang='pt-br').translate(word)
            dictionary.setdefault(word, [translation])

        except NotTranslated:
            dictionary.setdefault(word, [word])

    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df = df.T
    df.to_csv('en-pt1.csv', index=False)
    return df


def main():
    en, pt = create_dataset(PATH, 30000, 0)
    print(en[100])
    print(pt[100])

    translator = EATranslator(pt, en[5000], pt[5000])
    translator.run_evaluations()


if __name__ == '__main__':
    main()
