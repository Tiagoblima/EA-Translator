import itertools
import math
import re
import sys
import nltk
from util import get_vocabulary, create_dataset
from unidecode import unidecode

PATH = '../Resource/en-pt.csv'
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from collections import Counter
from itertools import combinations
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
import pandas as pd


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


def clean_text(text_list):
    stop_list = ['Article', 'Adverb', 'Noun', 'Adjective', 'Verb', r'\[', r'\]', 'Pronoun', r'[\n]', r'[\\]',
                 r'\([^()]*\)', r'\^', r'[//]']
    for i, text in enumerate(text_list):

        for regex in stop_list:
            text = unidecode(re.sub(regex, '', text).strip())

        text_list[i] = text
    return text_list


DIC_PATH = '../Resource/PORTUGUE.TXT'


def get_dic():
    df = pd.read_csv(PATH).dropna()
    df_dict = df.to_dict()
    dictionary = {}
    for key in df_dict.keys():
        dictionary[key] = list(df_dict[key].values())

    return dictionary


class EATranslator:
    """
        The class Evaluate the words to create Sentences
    """
    generation = None
    inp_vocabulary = None
    tar_vocabulary = None
    target_sentence = ''
    pmi = None
    selected_pairs = ()
    lang_dict = get_dic()
    found_sentence = None

    def __init__(self, sentences_list, input_sentence, target_sentence):
        """
            Here we obtain the vocabulary and the input and target sentence for translation
        :param sentences_list: list of sentences of the target language
        :param input_sentence: The input sentence
        :param target_sentence: target sentence
        """

        self.inp_vocabulary = input_sentence.split()
        self.tar_vocabulary = get_vocabulary(sentences_list)
        self.target_sentence = target_sentence
        self.pmi = PMI(sentences_list).get_all_pmi()
        self.generation = list(self.pmi.items())
        self.words_pmis = PMI(sentences_list).get_all_pmi()
        self.found_sentence = []

    def initialize_population(self):

        """
            Initialize the first population selecting the words in the dictionary look up.
        :return: The first generation
        """
        self.generation = []
        for word in self.inp_vocabulary:
            try:

                self.generation.append(self.lang_dict[word])

            except KeyError:
                pass
        assert len(self.generation) >= 1, "None of the words were found in the dictionary."

    def run_evaluations(self):

        """
            Run the evaluations of each word in the vocabulary to add it or not to the
            next generation of 'parents' word
        :return:
        """

        bi_gram = []
        i = 0

        self.initialize_population()
        print("target sentence: ", self.target_sentence.split(' '))
        print("Vocabulary: ", self.tar_vocabulary)

        parents = list(itertools.chain.from_iterable(self.generation))
        print("Parents: ", parents)

        while len(self.target_sentence.split(' ')) > i:

            evaluation_array = []
            for word in self.tar_vocabulary:

                if sys.intern(word) is not sys.intern(parents[-1]):
                    parents.append(word)
                    bi_gram.append(word)
                    # Markov Hidden states
                    fitness = bleu(hypothesis=parents, references=[self.target_sentence.split(' ')],
                                   auto_reweigh=True)

                    evaluation_array.append((word, fitness))
                    parents = parents[:-1]
                bi_gram = []
            evaluation_array.sort(key=lambda tup: tup[-1], reverse=True)

            i += 1
            new_word = evaluation_array[0][0]
            if new_word not in parents:
                parents.append(new_word)
            else:
                self.tar_vocabulary.remove(new_word)

        self.found_sentence = parents

        self.organize_words()

    def organize_words(self):

        """
         After run all the generation, we organize the sentence finding the first word in the original sentence
         and through blue evaluation finding the words after it.
        :return:
        """
        # -------- Finds the first word -------- #
        word_fit = []
        
        for word in self.found_sentence:
            fitness = bleu(hypothesis=word, references=self.target_sentence.split(' ')[:1],
                           auto_reweigh=True)
            word_fit.append((word, fitness))

        word_fit.sort(key=lambda tup: tup[1], reverse=True)

        first_word = word_fit[0][0]
        # -------------------------------------------

        self.found_sentence.remove(first_word)

        # ----- search for the next words -------------#
        final_sentence = [first_word]
        bi_gram = [first_word]
        fitness = []
        i = 1

        while len(self.target_sentence.split(' ')) > i:
            i += 1
            for word in self.found_sentence:
                if sys.intern(word) is not sys.intern(bi_gram[0]):
                    bi_gram.append(word)
                    bi_fitness = bleu(hypothesis=bi_gram, references=[self.target_sentence.split(' ')],
                                      auto_reweigh=True)
                    bi_gram = bi_gram[:-1]
                    fitness.append((word, bi_fitness))

            fitness.sort(key=lambda tup: tup[-1], reverse=True)
            final_sentence.append(fitness[0][0])
            bi_gram = [fitness[0][0]]
            fitness = []

        last_fitness = bleu(hypothesis=final_sentence, references=[self.target_sentence.split(' ')],
                            auto_reweigh=True)

        print("Last generation: {}\nLast Fitness {}".format(final_sentence, last_fitness))
