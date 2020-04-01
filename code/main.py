import math
import sys

import nltk

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from util import create_dataset, get_vocabulary
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
    vocabulary = [word for sentence in raw_text for word in sentence.split(' ')]
    word_count = Counter(vocabulary)
    num_words = len(vocabulary)
    return dict([(word, count / num_words) for word, count in word_count.items()])


def get_bi_grams_count(raw_text):
    """
     Generate the possibles bi_gram of words and its relative count inside the raw text
    :param raw_text: A list of sentences
    :return: a dictionary of big_grams where keys:bi_grams and values:relative count
    """
    bi_grams = []
    for sentence in raw_text:
        bi_grams.extend(list(nltk.bigrams(sentence.split())))
    num_bi_grams = len(bi_grams)
    bi_gram_count = dict([(bi_gram, count / num_bi_grams) for bi_gram, count in Counter(bi_grams).items()])

    return bi_gram_count


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
        p_xy = self.bi_grams_count[pair]
        return math.log(p_xy / (p_x * p_y), 2)

    def get_all_pmi(self):
        """
         Calculates all the possible Point-wise Mutual Information of the words bi_grams
        :return: A dictionary of key:bi_gram value: Point-wise Mutual Information associated
        """
        self.all_pmi = [(bi_gram, self.get_pmi(bi_gram)) for bi_gram in self.bi_grams_count.keys()
                        if self.get_pmi(bi_gram) > 0]

        return dict(self.all_pmi)


# ref = 'semeia-se corpo animal , ressucitara corpo esperitual . se ha corpo animal , ha tambem corpo esperitual'
# test_bleu = bleu(hypothesis=sentence_pt.split(' '), references=[sentence_pt.split(' '), ref.split(' ')],
# auto_reweigh=True)
def swap_position(bi_tuple):
    bi_tuple[0], bi_tuple[1] = bi_tuple[1], bi_tuple[0]
    return bi_tuple


class EATranslator:
    """
        The class Evaluate the words to create Sentences
    """
    generation = None
    vocabulary = None
    target_sentence = ''
    pmi = None
    selected_pairs = ()

    def __init__(self, sentences_list, target_sentence):
        self.vocabulary = get_vocabulary(sentences_list)
        self.target_sentence = target_sentence
        self.pmi = PMI(sentences_list).get_all_pmi()
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
         Select the pairs for the next generation based on the Point-Wise Mutual Information. Words with such information
         have the chance to reach the next generation.
        :return:
        """

        pairs = combinations(self.generation, 2)
        next_generation = []
        for pair in pairs:

            try:
                pmi = self.words_pmis[pair]
                next_generation.append((pair, pmi))
            except KeyError:
                try:
                    pmi = self.words_pmis[tuple(swap_position(list(pair)))]
                    next_generation.append((tuple(swap_position(list(pair))), pmi))
                except KeyError:
                    pass

        self.generation = next_generation

    def run_evaluations(self):
        self.initialize_population()

        self.select_pairs()

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
        """"""
        print(parents)
        while len(self.target_sentence.split(' ')) > i:

            evaluation_array = []
            for word in self.vocabulary:
                if sys.intern(word) is not sys.intern(parents[-1]):
                    parents.append(word)
                    fitness = bleu(hypothesis=parents, references=[self.target_sentence.split(' ')],
                                   auto_reweigh=True)
                    parents = parents[:-1]
                    evaluation_array.append((word, fitness))

            evaluation_array.sort(key=lambda tup: tup[-1], reverse=True)
            i += 1
            parents.append(evaluation_array[0][0])

        last_fitness = bleu(hypothesis=parents, references=[self.target_sentence.split(' ')],
                            auto_reweigh=True)
        print("Last generation: {}\nLast Fitness {}".format(parents, last_fitness))


def main():
    gu, pt = create_dataset(PATH, 1000)
    print(pt[1])
    eatranslator = EATranslator(pt, pt[1])
    eatranslator.run_evaluations()


if __name__ == '__main__':
    main()
