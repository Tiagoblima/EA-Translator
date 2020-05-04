import io
import re
import unicodedata
from nltk import RegexpTokenizer
from pysinonimos.sinonimos import Search
from textblob import TextBlob
from textblob.exceptions import NotTranslated
import pandas as pd
PATH = '../Resource/por.txt'


class Stack:
    stack = None

    def __init__(self):
        self.stack = []

    def push(self, element):
        self.stack.append(element)

    def pop(self):
        return self.stack.pop()

    def is_empty(self):
        if len(self.stack) is 0:
            return True

        return False


class Queue:
    queue = None

    def __init__(self):
        self.queue = []

    def push(self, element):
        self.queue.append(element)

    def pop(self):
        return self.queue.pop(0)

    def is_empty(self):
        if len(self.queue) is 0:
            return True

        return False


class PriorityQueue:
    priority_queue = None

    def __init__(self):
        self.priority_queue = []

    def push(self, element, heuristic):
        self.priority_queue.append((element, heuristic))
        self.priority_queue.sort(key=lambda tup: tup[1], reverse=True)

    def pop(self):
        return self.priority_queue.pop()

    def is_empty(self):
        if len(self.priority_queue) is 0:
            return True

        return False


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    tokenizer = RegexpTokenizer(r'\w+')

    return ' '.join(tokenizer.tokenize(w)).lower()


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples, start=5000):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:-1]] for l in lines[start:
                                                                                      start + num_examples]]

    return zip(*word_pairs)


def get_vocabulary(text):
    var = [word for sentence in text for word in sentence.split(' ')]

    return set(var)



