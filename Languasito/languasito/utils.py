import re
from torch.utils.data import Dataset
import json, os
from tqdm.autonotebook import tqdm as tqdm
import torch
import numpy as np
import random
import pickle


class LanguasitoTokenizer:
    def __init__(self, no_space_language=False):
        self._no_space_language = no_space_language

    def __call__(self, text):
        if self._no_space_language:
            return [ch for ch in text]
        else:
            toks = []
            tok = ''
            for ch in text:
                if not ch.isalnum() or ch == ' ':
                    tok = tok.strip()
                    if len(tok) != 0:
                        toks.append(tok)
                        tok = ''
                    if ch != ' ':
                        toks.append(ch)
                else:
                    tok += ch
            if tok.strip() != '':
                toks.append(tok)

            return toks
    # def __call__(self, text):
    #     if self._no_space_language:
    #         return [ch for ch in text]
    #     else:
    #         punctuation = '''"’'()[]{}<>:,‒–—―…!.«»-?‘’“”;/⁄␠·&@*\\•^¤¢$€£¥₩₪†‡°¡¿¬#№%‰‱¶′§~¨_|¦⁂☞∴‽※"„”'''
    #         new_text = ''
    #         for ch in text:
    #             if re.match(u'[\u4e00-\u9fff]', ch):
    #                 new_text += ' ' + ch + ' '
    #             elif ch in punctuation:
    #                 new_text += ' ' + ch + ' '
    #             else:
    #                 new_text += ch
    #
    #         tmp = new_text.replace('  ', ' ')
    #         while tmp != new_text:
    #             new_text = tmp
    #             tmp = new_text.replace('  ', ' ')
    #         new_text = new_text.strip()
    #         return new_text.split(' ')


def mp_job(data):
    no_space_lang, lines = data
    print(f"\t\ttokenizing {len(lines)} ...")
    _st = LanguasitoTokenizer(no_space_language=no_space_lang)
    filtered_lines = []
    # new_lines = []
    for line in lines:
        toks = _st(line)
        if len(toks) > 5 and len(toks) < 50:
            valid = True
            for tok in toks:
                if len(tok) > 20:
                    valid = False
                    break
            if valid:
                filtered_lines.append(toks)
                # new_lines.append(line)

    return filtered_lines


class LanguasitoDataset(Dataset):
    def __init__(self, no_space_lang=False):
        self._examples = []
        self._st = LanguasitoTokenizer(no_space_language=no_space_lang)
        self.no_space_lang = no_space_lang

    def load_file(self, filename: str):
        print(f"Loading {filename}")

        if os.path.exists(filename + ".pickle"):
            print("\tloading from cached file ...")
            self._examples = pickle.load(open(filename + ".pickle", "rb"))
            print(f"\tdataset has {len(self._examples)} lines.")
            return

        import multiprocessing
        lines = []
        chunks = []
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                l = line.strip()
                if l == "":
                    continue
                lines.append(l)
                if len(lines) > 999999:  # 1M
                    chunks.append(lines)
                    lines = []
                    print(f"\treading chunk #{len(chunks)} ...")
                if len(chunks) > 100:  # 200 M lines
                    break
            if len(lines) > 0:
                chunks.append(lines)

        cpu_count = int(multiprocessing.cpu_count() / 2)
        print(f"\tloaded {len(chunks)} chunks, now filtering on {cpu_count} threads ...")

        packed_chunks = [(self.no_space_lang, lines) for lines in chunks]

        p = multiprocessing.Pool(processes=cpu_count)
        return_data = p.map(mp_job, packed_chunks)
        p.close()
        p.join()

        cnt = 0
        for lines in return_data:
            for line in lines:
                self._examples.append([line, cnt])
                cnt += 1

        """
        filtered_lines, o_lines = self._filter(lines)
        n = len(filtered_lines)
        for ii in range(n):
            tokenized = filtered_lines[ii]
            if o_lines[ii].startswith("<doc id="):
                continue
            if ii < n - 1 and not o_lines[ii + 1].startswith("<doc id="):
                ni = len(self._examples) + 1
            else:
                ni = len(self._examples) - 1
            self._examples.append([tokenized, ni])
        """
        print(f"\tdataset has {len(self._examples)} lines.")
        pickle.dump(self._examples, open(filename + ".pickle", "wb"))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        sent1 = self._examples[item][0]
        sent2 = self._examples[self._examples[item][1]][0]
        return {'sent1': sent1, 'sent2': sent2}

    def _filter(self, lines):
        filtered_lines = []
        new_lines = []
        for line in lines:
            toks = self._st(line)
            if len(toks) > 5 and len(toks) < 50:
                valid = True
                for tok in toks:
                    if len(tok) > 20:
                        valid = False
                        break
                if valid:
                    filtered_lines.append(toks)
                    new_lines.append(line)
        return filtered_lines, new_lines


class LangusitoWordDecomposer:
    def __init__(self):
        self._tok2int = {}

    def train(self, dataset: LanguasitoDataset, w_cutoff=7, max_vocab_size=1000, max_ngram=-1):
        word2count = {}
        n = len(dataset)
        for ii in tqdm(range(n), ncols=100, desc='Updating encodings'):
            tokenized = dataset[ii]['sent1']
            for tok in tokenized:
                if tok in word2count:
                    word2count[tok] += 1
                else:
                    word2count[tok] = 1

        word_list = []
        for word in word2count:
            if word2count[word] > w_cutoff:
                word_list.append(word)
                for c in word:
                    if c not in self._tok2int:
                        self._tok2int[c] = len(self._tok2int)

        ngram2count = {}
        for word in word_list:
            # get all ngrams
            ngrams = self._extract_ngrams(word, max_ngram)
            for ngram in ngrams:
                if ngram in ngram2count:
                    ngram2count[ngram] += 1
                else:
                    ngram2count[ngram] = 1

        order2count = {}
        for ngram in ngram2count:
            key = len(ngram)
            if key not in order2count:
                order2count[key] = ngram2count[ngram]
            else:
                order2count[key] += ngram2count[ngram]

        sorted_ngrams = [k for k, v in sorted(ngram2count.items(), reverse=True,
                                              key=lambda item: item[1])]
        for ngram in sorted_ngrams[:min(len(sorted_ngrams), max_vocab_size - len(self._tok2int))]:
            self._tok2int[ngram] = len(self._tok2int)

    @staticmethod
    def _extract_ngrams(word, max_len):
        if max_len == -1:
            max_len = len(word)
        max_size = min(max_len, len(word))

        ngrams = []
        for ii in range(1, max_size):
            for jj in range(len(word) - ii):
                ngrams.append(word[jj:jj + ii + 1])
        return ngrams

    @staticmethod
    def _find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path

        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = LangusitoWordDecomposer._find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def _build_graph(self, word):
        graph = {}
        for ii in range(len(word) - 1):
            start_node = ii
            graph[start_node] = [start_node + 1]
            for jj in range(ii + 2, len(word) + 1):
                end_node = jj
                tok = word[ii:jj]
                if tok in self._tok2int:
                    graph[start_node].append(end_node)
        graph[len(word) - 1] = [len(word)]
        graph[len(word)] = []
        return graph

    def tokenize(self, word_list: list):
        tokenized = []
        for word in word_list:
            graph = self._build_graph(word)
            spath = LangusitoWordDecomposer._find_shortest_path(graph, 0, len(word))
            toks = []
            for ii in range(1, len(spath)):
                toks.append(word[spath[ii - 1]:spath[ii]])
            tokenized.append(toks)
        return tokenized


def load_dataset(filename: str) -> LanguasitoDataset:
    dataset = LanguasitoDataset()
    print(f"Reading dataset file {filename} ...")
    lines = open(filename, "r", encoding="utf8").readlines()
    for ii in tqdm(range(len(lines)), desc='Loading dataset "{0}"'.format(filename), ncols=100):
        fname = lines[ii].strip()
        dataset.load_file(fname)
    return dataset


def mask_concat(representations, drop_prob: float, training: bool, device: str):
    if training:
        masks = []
        for ii in range(len(representations)):
            mask = np.ones((representations[ii].shape[0], representations[ii].shape[1]), dtype=np.long)
            masks.append(mask)

        for ii in range(masks[0].shape[0]):
            for jj in range(masks[0].shape[1]):
                mult = 1
                for kk in range(len(masks)):
                    p = random.random()
                    if p < drop_prob:
                        mult += 1
                        masks[kk][ii, jj] = 0
                for kk in range(len(masks)):
                    masks[kk][ii, jj] *= mult
        for kk in range(len(masks)):
            masks[kk] = torch.tensor(masks[kk], device=device)

        for kk in range(len(masks)):
            representations[kk] = representations[kk] * masks[kk].unsqueeze(2)

    return torch.cat(representations, dim=-1)


class Encodings:
    def __init__(self, max_vocab_size: int = 10000, min_word_occ: int = 5, min_char_occ: int = 20):
        self._max_vocab_size = max_vocab_size
        self._min_word_occ = min_word_occ
        self._min_char_occ = min_char_occ
        self.char2int = {'<PAD>': 0, '<UNK>': 1, '<SOT>': 2, '<EOT>': 3}
        self.word2int = {}
        self.char_list = []
        self.word_decomposer = LangusitoWordDecomposer()

    def load(self, filename: str):
        json_obj = json.load(open(filename))
        self.char2int = json_obj['char2int']
        if 'word2int' in json_obj:
            self.word2int = json_obj['word2int']

        self.char_list = [' ' for _ in range(len(self.char2int))]
        for char in self.char2int:
            self.char_list[self.char2int[char]] = char

        self.word_decomposer._tok2int = json_obj['tok2int']

    def save(self, filename: str, full: bool = True):
        json_obj = {'char2int': self.char2int}
        json_obj['tok2int'] = self.word_decomposer._tok2int
        if full:
            json_obj['word2int'] = self.word2int
        json.dump(json_obj, open(filename, 'w'))

    def update(self, dataset: LanguasitoDataset):
        self.word_decomposer.train(dataset)
        word2count = {}
        char2count = {}
        n = len(dataset)
        for ii in tqdm(range(n), ncols=100, desc='Updating encodings'):
            tokenized = dataset[ii]['sent1']
            for tok in tokenized:
                tok_lower = tok.lower()
                if tok_lower not in word2count:
                    word2count[tok_lower] = 1
                else:
                    word2count[tok_lower] += 1

                for ch in tok_lower:
                    if ch not in char2count:
                        char2count[ch] = 1
                    else:
                        char2count[ch] += 1

        # sort dict
        sorted_words = [k for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True)]
        sorted_words = sorted_words[:min(len(sorted_words), self._max_vocab_size)]
        for w in sorted_words:
            if word2count[w] > self._min_word_occ:
                self.word2int[w] = len(self.word2int)

        for ch in char2count:
            if char2count[ch] > self._min_char_occ:
                self.char2int[ch] = len(self.char2int)

        self.char_list = [' ' for _ in range(len(self.char2int))]
        for char in self.char2int:
            self.char_list[self.char2int[char]] = char


class LanguasitoCollate:
    def __init__(self, encodings: Encodings, live: bool = False):
        self._encodings = encodings
        self._live = live

    def collate_fn(self, batch):
        if not self._live:
            new_batch = []
            for b in batch:
                new_batch.append(b['sent1'])
                new_batch.append(b['sent2'])
            batch = new_batch
        a_sent_len = [len(sent) for sent in batch]
        a_word_len = []
        word_list = []
        for sent in batch:
            for word in sent:
                a_word_len.append(len(word))
                word_list.append(word)
        x_sent_len = np.array(a_sent_len, dtype=np.long)
        x_word_len = np.array(a_word_len, dtype=np.long)
        max_sent_len = np.max(x_sent_len)
        max_word_len = np.max(x_word_len)
        x_sent_masks = np.zeros((len(batch), max_sent_len), dtype=np.float)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.float)

        x_word_char = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.long)
        word_targets = self._encodings.word_decomposer.tokenize(word_list)
        max_word_dec = max([len(w) for w in word_targets])
        x_word_decoder = np.zeros((x_word_len.shape[0], max_word_dec + 2), dtype=np.long)
        c_word = 0
        x_lang_sent = np.zeros((len(batch)), dtype=np.long)
        x_lang_word = []

        for iSent in range(len(batch)):
            sent = batch[iSent]
            x_lang_sent[iSent] = 1
            for iWord in range(len(sent)):
                x_word_char[iWord, 0] = 2  # start of token
                word = sent[iWord]
                x_sent_masks[iSent, iWord] = 1
                x_lang_word.append(1)
                target = word_targets[c_word]
                x_word_decoder[c_word, 0] = 2
                for iTarget in range(len(target)):
                    tgt = target[iTarget]
                    if tgt in self._encodings.word_decomposer._tok2int:
                        x_word_decoder[c_word, iTarget + 1] = self._encodings.word_decomposer._tok2int[tgt] + 4
                x_word_decoder[c_word, len(target) + 1] = 3

                for iChar in range(len(word)):
                    x_word_masks[c_word, iChar] = 1
                    ch = word[iChar]
                    if ch.lower() == ch.upper():  # symbol
                        x_word_case[c_word, iChar] = 1
                    elif ch.lower() != ch:  # upper
                        x_word_case[c_word, iChar] = 2
                    else:  # lower
                        x_word_case[c_word, iChar] = 3
                    ch = ch.lower()
                    if ch in self._encodings.char2int:
                        x_word_char[c_word, iChar + 1] = self._encodings.char2int[ch]
                    else:
                        x_word_char[c_word, iChar + 1] = self._encodings.char2int['<UNK>']
                x_word_char[c_word, len(word) + 1] = 3  # end of token
                x_word_masks[c_word, len(word) + 1] = 1
                c_word += 1

        x_lang_word = np.array(x_lang_word)
        response = {
            'x_word_char': torch.tensor(x_word_char),
            'x_word_case': torch.tensor(x_word_case),
            'x_lang_word': torch.tensor(x_lang_word),
            'x_sent_len': torch.tensor(x_sent_len),
            'x_word_len': torch.tensor(x_word_len),
            'x_sent_masks': torch.tensor(x_sent_masks),
            'x_word_masks': torch.tensor(x_word_masks),
            'x_word_targets': torch.tensor(x_word_decoder),
            'x_max_len': max_sent_len
        }

        return response
