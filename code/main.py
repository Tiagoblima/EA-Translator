import itertools
import math
import re
from random import shuffle

from util import create_dataset, get_vocabulary
from nltk import ngrams
from collections import Counter
from itertools import combinations
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

PATH = '../Resource/guarani-portugues.txt'


def get_word_count(raw_text):
    """
     Generate the relative count of the words
    :param raw_text: A list of sentence
    :return: a dictionary of words where the key:word and value:relative count
    """
    word_list = [word for sentence in raw_text for word in sentence.split(' ')]
    word_count = Counter(word_list)
    num_words = len(word_list)
    return dict([(word, count / num_words) for word, count in word_count.items()])


def get_bi_grams_count(raw_text):
    """
     Generate the possibles bi_gram of words and its relative count inside the raw text
    :param raw_text: A list of sentences
    :return: a dictionary of big_grams where keys:bi_grams and values:relative count
    """
    bi_grams = itertools.chain.from_iterable([combinations(sentence.split(' '), 2) for sentence in raw_text])
    bi_grams = [' '.join(bi_gram) for bi_gram in bi_grams]
    bi_gram_count = Counter(bi_grams)
    num_bi_grams = len(bi_grams)
    return dict([(bi_gram, count / num_bi_grams) for bi_gram, count in bi_gram_count.items()])


class PMI:
    """
        Select the best related pairs of words
    """
    word_count = None
    bi_grams_count = None
    all_pmi = list()

    def __init__(self, text_list):
        self.word_count = get_word_count(text_list)
        self.bi_grams_count = get_bi_grams_count(text_list)
        self.all_pmi = []

    def get_pmi(self, pair=()):
        """
         Generate the Point-wise Mutual Information to obtain the correlation between the words in the corpus.
        :param pair: pair of words
        :return: math.log(p_xy / (p_x * p_y), 2), where p_xy is the probability of the words appear together and
        p_x and p_y are the probability of both words appear alone respectively.
        """
        x = pair[0]
        y = pair[1]
        p_x = self.word_count[x]
        p_y = self.word_count[y]
        p_xy = self.bi_grams_count[' '.join(pair)]
        return math.log(p_xy / (p_x * p_y), 2)

    def get_all_pmi(self):
        """
         Calculates all the possible Point-wise Mutual Information of the words bi_grams
        :return: A dictionary of key:bi_gram value: Point-wise Mutual Information associated
        """
        self.all_pmi = [(bi_gram, self.get_pmi(bi_gram.split(' '))) for bi_gram in self.bi_grams_count.keys()]

        return dict([pmi_p for pmi_p in self.all_pmi if pmi_p[-1] > 0])


# ref = 'semeia-se corpo animal , ressucitara corpo esperitual . se ha corpo animal , ha tambem corpo esperitual'
# test_bleu = bleu(hypothesis=sentence_pt.split(' '), references=[sentence_pt.split(' '), ref.split(' ')],
# auto_reweigh=True)

class EATranslator:
    """
        The class Evaluate the words to create Sentences
    """
    generation = None
    vocabulary = None
    target_sentence = ''
    words_pmis = dict()
    selected_pairs = ()

    def __init__(self, sentences_list, target_sentence):
        self.vocabulary = get_vocabulary(sentences_list)
        self.target_sentence = target_sentence
        self.words_pmis = PMI(sentences_list).get_all_pmi()

    def initialize_population(self):

        """
            Initialize the first population selecting the best words which the bleu score is not zero
        :return: The first generation
        """
        word_bleu = []
        for word in self.vocabulary:
            test_bleu = bleu(hypothesis=[word], references=[self.target_sentence.split(' ')],
                             auto_reweigh=True)
            word_bleu.append((word, test_bleu))

        self.generation = [tup[0] for tup in word_bleu if tup[1] is not 0]

    def select_pairs(self):

        """
         Select the pairs for the next generation
        :return:
        """

        pairs = combinations(self.generation, 2)
        next_generation = []
        for pair in pairs:

            try:
                pmi = self.words_pmis[' '.join(pair)]
                next_generation.append((pair, pmi))
            except KeyError:
                pass

        self.generation = next_generation

    def run_evaluations(self):
        self.initialize_population()
        self.select_pairs()
        print(self.generation)
        next_generation = [tup[0][0] for tup in self.generation]
        self.vocabulary = [tup[0][1] for tup in self.generation]
        self.vocabulary.extend(next_generation)

        self.vocabulary = set(self.vocabulary)

        # Select the first best parents
        evaluation_array = []
        for n_gram in self.generation:
            test_bleu = bleu(hypothesis=n_gram[0], references=[self.target_sentence.split(' ')[:2]],
                             auto_reweigh=True)

            evaluation_array.append((n_gram[0], test_bleu))

        evaluation_array.sort(key=lambda tup: tup[-1], reverse=True)
        parents = list(evaluation_array[0][0])
        # ------------------------------------------------------------

        i = 0
        last_fitness = 0
        # Search for the best children
        smooth_function = SmoothingFunction()
        while len(self.target_sentence.split(' ')) > i:
            for word in self.vocabulary:
                parents.append(word)
                test_bleu = bleu(hypothesis=parents, references=[self.target_sentence.split(' ')],
                                 auto_reweigh=True)
                evaluation_array.append((word, test_bleu))
                parents = parents[:-1]

            evaluation_array.sort(key=lambda tup: tup[-1], reverse=True)
            i += 1
            parents.append(evaluation_array[0][0])

        last_fitness = bleu(hypothesis=parents, references=[self.target_sentence.split(' ')],
                            auto_reweigh=True)
        print("Last generation: {}\nLast Fitness {}".format(parents, last_fitness))


def main():
    gu, pt = create_dataset(PATH, 1000)

    selection = PMI(pt)
    pmis = selection.get_all_pmi()
    print(pt[0])
    eatranslator = EATranslator(pt, pt[0])
    eatranslator.run_evaluations()


if __name__ == '__main__':
    main()
