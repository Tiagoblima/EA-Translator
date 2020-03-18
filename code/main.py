import itertools
import math
import re
from random import shuffle

from util import create_dataset, get_vocabulary
from nltk import ngrams
from collections import Counter
from itertools import combinations
from nltk import bleu

PATH = '../Resource/guarani-portugues.txt'


def get_word_count(text_list):
    word_list = [word for sentence in text_list for word in sentence.split(' ')]
    word_count = Counter(word_list)
    num_words = len(word_list)
    return dict([(word, count / num_words) for word, count in word_count.items()])


def get_bi_grams_count(text_list):
    bi_grams = itertools.chain.from_iterable([combinations(sentence.split(' '), 2) for sentence in text_list])
    bi_grams = [' '.join(bi_gram) for bi_gram in bi_grams]
    bi_gram_count = Counter(bi_grams)
    num_bi_grams = len(bi_grams)
    return dict([(bi_gram, count / num_bi_grams) for bi_gram, count in bi_gram_count.items()])


class Selection:
    word_count = None
    bi_grams_count = None
    all_pmi = list()

    def __init__(self, text_list):
        self.word_count = get_word_count(text_list)
        self.bi_grams_count = get_bi_grams_count(text_list)
        self.all_pmi = []

    def get_pmi(self, pair=()):
        x = pair[0]
        y = pair[1]
        p_x = self.word_count[x]
        p_y = self.word_count[y]
        p_xy = self.bi_grams_count[' '.join(pair)]
        return math.log(p_xy / (p_x * p_y), 2)

    def get_all_pmi(self):
        self.all_pmi = [(bi_gram, self.get_pmi(bi_gram.split(' '))) for bi_gram in self.bi_grams_count.keys()]
        self.all_pmi.sort(reverse=True)
        return dict(self.all_pmi)


# ref = 'semeia-se corpo animal , ressucitara corpo esperitual . se ha corpo animal , ha tambem corpo esperitual'
# test_bleu = bleu(hypothesis=sentence_pt.split(' '), references=[sentence_pt.split(' '), ref.split(' ')],
# auto_reweigh=True)

class EATranslator:
    generation = None
    vocabulary = None
    target_sentence = ''
    words_pmis = dict()
    selected_pairs = ()

    def __init__(self, sentences_list, target_sentence):
        self.vocabulary = get_vocabulary(sentences_list)
        self.target_sentence = target_sentence
        self.words_pmis = Selection(sentences_list).get_all_pmi()

    def initialize_gen(self):
        word_bleu = []
        for word in self.vocabulary:
            test_bleu = bleu(hypothesis=[word], references=[self.target_sentence.split(' ')],
                             auto_reweigh=True)
            word_bleu.append((word, test_bleu))

        self.generation = [tup[0] for tup in word_bleu if tup[1] is not 0]

    def pairwise_selection(self):
        pairs = combinations(self.generation, 2)
        pmi_pair = []
        for pair in pairs:

            try:
                pmi = self.words_pmis[' '.join(pair)]
                pmi_pair.append((pair, pmi))
            except:
                pass

        pmi_pair.sort(key=lambda tup: tup[-1], reverse=True)
        self.generation = [pmi_p for pmi_p in pmi_pair if pmi_p[-1] > 0]

    def run_evaluations(self):
        self.initialize_gen()
        self.pairwise_selection()
        evaluation_array = []
        next_generation = [tup[0][0] for tup in self.generation]
        self.vocabulary = [tup[0][1] for tup in self.generation]
        self.vocabulary.extend(next_generation)

        self.vocabulary = set(self.vocabulary)
        print(self.vocabulary)
        new_n_grams = self.generation[0][0]
        for n_gram in new_n_grams:

            test_bleu = bleu(hypothesis=new_n_grams, references=[self.target_sentence.split(' ')],
                             auto_reweigh=True)

            evaluation_array.append((n_gram[0], test_bleu))

        evaluation_array.sort(key=lambda tup: tup[-1], reverse=True)
        print(evaluation_array)


def main():
    gu, pt = create_dataset(PATH, 1000)

    selection = Selection(pt)
    pmis = selection.get_all_pmi()
    eatranslator = EATranslator(pt, pt[0])
    eatranslator.run_evaluations()


if __name__ == '__main__':
    main()
