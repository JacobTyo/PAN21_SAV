import string
from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers.models.tokenizer import WordTokenizer
import torch
from transformers import AutoTokenizer


class CharTokenizer(WordTokenizer):

    vocab = dict([(c, i) for i, c in enumerate(string.printable)])

    def set_vocab(self, vocab: Iterable[str]):
        # the vocab we are going to use is just the list of all chars.
        # this shouldn't do anythign I believe. . .
        self.vocab = list(vocab)

    def get_vocab(self, vocab: Iterable[str]):
        if vocab is None:
            return self.vocab
        else:
            assert False, 'idk why this is here, but I am using it and this shouldnt happen'

    def tokenize(self, text: str) -> List[int]:
        # work around maybe?
        if len(text) > 10000:
            text = text[:10000]

        tokens = []
        for c in text:
            try:
                tokens.append(self.vocab[c])
            except:
                tokens.append(self.vocab['~'])

        return tokens

    def save(self, output_path: str):
        # what am I saving here? nothing is important I think, this is a static optimizer
        print('the save function of my char tokenizer was called, but nothing to do. . . ')

    @staticmethod
    def load(input_path: str):
        print('the load function of my char tokenizer was called, but nothing to do. . . ')


class MyTokenizerFromTransformers(WordTokenizer):
    '''
    just a class to transform a tokenizer from the huggingface transformers library to a
    tokenizer for use with sentense_transformers
    '''

    def __init__(self, transformers_tokenizer):
        super(MyTokenizerFromTransformers).__init__()
        self.tokenizer = transformers_tokenizer

    def set_vocab(self, vocab):
        # nothign to do here
        pass

    def get_vocab(self, vocab):
        # sure
        return self.tokenizer.vocab

    def tokenize(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def save(self, output_path):
        # I should implement this if I am updating embeddings, else not needed.
        pass

    @staticmethod
    def load(input_path):
        # need this, again, only if using updating embeddings
        pass