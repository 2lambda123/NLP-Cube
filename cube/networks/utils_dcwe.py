import sys
import random
from typing import *

sys.path.append('')
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from cube.io_utils.objects import Document, Sentence, Token, Word
from cube.io_utils.encodings import Encodings

from collections import namedtuple


class DCWEDataset(Dataset):
    def __init__(self):
        """Initializes the examples list.
        Parameters:
            - None
        Returns:
            - None
        Processing Logic:
            - Creates an empty list.
            - No parameters are passed.
            - No return value is expected.
            - Used to initialize the examples list."""
        
        self._examples = []

    def __len__(self):
        """This function returns the length of the examples attribute.
        Parameters:
            - self (class object): The class object itself.
        Returns:
            - int: The length of the examples attribute.
        Processing Logic:
            - Returns the length of the examples attribute."""
        
        return len(self._examples)

    def __getitem__(self, item):
        """Returns the item at the given index.
        Parameters:
            - item (int): Index of the item to be returned.
        Returns:
            - type: The item at the given index.
        Processing Logic:
            - Returns the item at the given index.
            - Uses the self._examples list.
            - Index must be an integer.
            - Raises an IndexError if index is out of range."""
        
        return self._examples[item]

    def load_language(self, filename: str, lang: str):
        """Loads language data from a file and adds it to the examples list.
        Parameters:
            - filename (str): The name of the file to load data from.
            - lang (str): The language to load data for.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Open the file using the provided filename.
            - Read the first line of the file and split it by spaces.
            - Convert the first two elements of the split line to integers.
            - Loop through the number of examples.
            - Read the next line of the file and split it by spaces.
            - The first element is the word, the rest are the vector values.
            - Convert the vector values to floats.
            - Append the language, word, and vector to the examples list.
            - Close the file."""
        
        f = open(filename, encoding='utf-8')
        parts = f.readline(5_000_000).strip().split(' ')
        num_examples = int(parts[0])
        vector_len = int(parts[1])
        for ii in range(num_examples):
            parts = f.readline(5_000_000).strip().split(' ')
            word = parts[0]
            vector = [float(pp) for pp in parts[1:]]
            self._examples.append([lang, word, vector])
        f.close()


class DCWECollate:
    encodings: Encodings
    examples: List[Any]

    def __init__(self, encodings: Encodings):
        """"Initializes the class with the provided encodings and sets the start and stop values for the encodings."
        Parameters:
            - encodings (Encodings): A class containing character to integer encodings.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Sets the start and stop values.
            - Uses the length of the encodings to determine the start and stop values."""
        
        self.encodings = encodings
        self._start = len(encodings.char2int)
        self._stop = len(encodings.char2int) + 1

    def collate_fn(self, examples):
        """Purpose:
            This function takes in a list of examples and collates them into a dictionary of tensors for use in a neural network model.
        Parameters:
            - examples (list): A list of examples, where each example is a tuple containing the language, word, and vector representation.
        Returns:
            - collated (dict): A dictionary containing the collated tensors for the input examples.
        Processing Logic:
            - Creates empty lists for languages, vectors, and words.
            - Loops through each example and appends the language, word, and vector to their respective lists.
            - Calculates the maximum word length and creates empty arrays for the character, case, word length, mask, and language tensors.
            - Loops through each word and encodes the characters and case into the corresponding tensors.
            - Sets the language tensor to the corresponding integer value.
            - Returns a dictionary containing the collated tensors."""
        
        langs = []
        vectors = []
        words = []
        for example in examples:
            langs.append(example[0])
            words.append(example[1])
            vectors.append(example[2])

        max_word_len = max([len(word) for word in words]) + 2
        x_char = np.zeros((len(examples), max_word_len), dtype='long')
        x_case = np.zeros((len(examples), max_word_len), dtype='long')
        x_word_len = np.zeros((len(examples)), dtype='long')
        x_mask = np.ones((len(examples), 1))
        x_lang = np.ones((len(examples), 1))
        for ii in range(len(words)):
            word = words[ii]
            x_char[ii, 0] = self._start
            for jj in range(word):
                char = word[jj]
                ch_low = char.lower()
                if ch_low in self.encodings.char2int:
                    x_char[ii, jj + 1] = self.encodings.char2int[ch_low]
                else:
                    x_char[ii, jj + 1] = 1  # UNK
                if char.lower() == char.upper():
                    x_case[ii, jj + 1] = 1
                elif ch_low == char:
                    x_case[ii, jj + 1] = 2
                else:
                    x_case[ii, jj + 1] = 3

            x_char[len(word) + 1] = self._stop
            x_word_len[ii] = len(word)
            x_lang = self.encodings.lang2int[langs[ii]]

        collated = {'y_target': torch.tensor(np.array(vectors)),
                    'x_char': torch.tensor(x_char),
                    'x_case': torch.tensor(x_case),
                    'x_mask': torch.tensor(x_mask),
                    'x_lang': torch.tensor(x_lang),
                    'x_word_len': torch.tensor(x_word_len)}
