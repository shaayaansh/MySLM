import os
from pathlib import Path
import regex as re
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import parallelize_tokenize_file
from typing import Tuple, List
from collections import Counter
import unicodedata as ud

class BytePairEncodingTokenizer():
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.merges = []
        self.merge_ranks = {}
        self.b2u, self.u2b = self._bytes_to_unicode()
        self.token_to_id = {}
        self.id_to_token = []

    def _bytes_to_unicode(self):
        # visible ranges (don’t collide with space or control chars)
        bs = list(range(ord('!'), ord('~')+1)) + \
            list(range(ord('¡'), ord('¬')+1)) + \
            list(range(ord('®'), ord('ÿ')+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)  # map leftover bytes to safe code points
                n += 1
        b2u = {b: chr(c) for b, c in zip(bs, cs)}   # byte -> unicode char
        u2b = {v: k for k, v in b2u.items()}        # unicode char -> byte
        return b2u, u2b
    
    @classmethod
    def from_files(cls, vocab_path, merges_path):

        self = cls(data_path=None)

        vocab_path = Path(vocab_path)
        merges_path = Path(merges_path)

        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        if isinstance(vocab, dict):
            self.token_to_id = {str(k): int(v) for k, v in vocab.items()}
            size = 1 + max(self.token_to_id.values()) if self.token_to_id else 0
            self.id_to_token = [None] * size
            for tok, idx in self.token_to_id.items():
                if 0 <= idx < size and self.id_to_token[idx] is None:
                    self.id_to_token[idx] = tok
                else:
                    raise ValueError("vocab.json has non-contiguous or duplicate ids.")
                
            if any(t is None for t in self.id_to_token):
                raise ValueError("vocab.json has gaps in ids.")
                
        elif isinstance(vocab, list):
            self.id_to_token = [str(t) for t in vocab]
            self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
    
        else:
            raise TypeError("vocab file must be a dict or list.")
        

        merges = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(" ", 1) # split on first space
                if len(parts) != 2:
                    raise ValueError(f"Malformed merge line: {line}")
                
                a, b = parts[0], parts[1]
                merges.append((a, b))

        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}


        return self


    def initialize_vocabulary(self, special_tokens):
        self.token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token = special_tokens[:]
        
        # base byte tokens (each is a single printable Unicode "byte-char")
        for b in range(256):
            tok = self.b2u[b]
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

        self.merges = []

    def _add_merge(
            self,
            a: str, 
            b: str,
    ):
        """
        Add a new merge token a+b; update vocab; return (token, id)
        """
        new_token = a + b
        if new_token not in self.token_to_id:
            new_id = len(self.id_to_token)
            self.token_to_id[new_token] = new_id
            self.id_to_token.append(new_token)
            self.merges.append((a, b))
            self.merge_ranks[(a, b)] = len(self.merges) - 1
            return new_token, new_id
        
        # if merged token already present
        return new_token, self.token_to_id[new_token]

    @staticmethod
    def _find_all_pairs(s):
        return list(zip(s, s[1:]))

    def find_most_frequent_pair(self, token_counts: dict) -> list[tuple]:
        all_token_counts = Counter()
        for token, count in token_counts.items():
            if len(token) < 2:
                continue

            pairs = self._find_all_pairs(token)
            local_counts = Counter(pairs)

            all_token_counts.update({k: v * count for k, v in local_counts.items()})

        return all_token_counts.most_common(1)
    
    @staticmethod
    def _apply_merge_to_seq(seq, a, b, ab):
        out = []
        i = 0

        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i+1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(seq[i])
                i += 1
            
        return tuple(out)
        
    def train_bpe(
            self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple:
        # initialize the vocabulary
        self.initialize_vocabulary(special_tokens)
        print(f"Initial Vocabulary Length: {len(self.token_to_id)}")
        token_counts = parallelize_tokenize_file(input_path, desired_num_chunks=24, max_workers=8)
        
        corpus = {
            tuple(self.b2u[b] for b in ud.normalize("NFC", s).encode("utf-8")): freq
            for s, freq in token_counts.items()
        }
        
        target_merges = max(0, vocab_size - len(self.token_to_id))
        with tqdm(total=target_merges, dynamic_ncols=True, desc="Training BPE...") as pbar:
            while len(self.token_to_id) < vocab_size:
                pair = self.find_most_frequent_pair(corpus)[0][0]
                if pair is None:
                    break # no more mergable pairs; stop early

                a, b = pair    
                ab, _ = self._add_merge(a, b)
                new_corpus = {}
                for seq, freq in corpus.items():
                    new_seq = self._apply_merge_to_seq(seq, a, b, ab)
                    new_corpus[new_seq] = new_corpus.get(new_seq, 0) + freq

                corpus = new_corpus

                pbar.update(1)
                pbar.set_postfix_str(f"last merge: {len(ab)} chars")
        
        return self.token_to_id, self.merges
    
    def _encode_word(self, word: str) -> List[str]:
        symbols = [self.b2u[b] for b in word.encode("utf-8")]
        if not symbols:
            return []
        
        def best_pair(symbols):
            best = None
            best_rank = float('inf')

            for i in range(len(symbols)-1):
                pair = symbols[i], symbols[i+1]
                r = self.merge_ranks.get(pair)
                if r is not None and r < best_rank:
                    best_rank = r
                    best = (i, pair)

            return best
        
        while True:
            hit = best_pair(symbols)
            if hit is None:
                break
            else:
                i, (a, b) = hit
                symbols[i: i+2] = [a+b]

        return symbols
    
    def tokens_to_ids(self, tokens: List[str], strict: bool = False) -> List[int]:
        ids: List[int] = []
        for tok in tokens:
            tid = self.token_to_id.get(tok)
            if tid is not None:
                ids.append(tid)
            elif strict:
                raise KeyError(f"Unknown token: {tok!r}")
            else:
                for ch in tok:
                    base_id = self.token_to_id.get(ch)
                    if base_id is None:
                        raise KeyError(f"Missing base byte token for {ch!r}")
                    ids.append(base_id)

        return ids
    

    def encode(self, text: str, return_str_tokens: bool = False):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        str_tokens: List[str] = []
        for word in re.findall(PAT, text):
            str_tokens.extend(self._encode_word(word))

        ids = self.tokens_to_ids(str_tokens, strict=False)

        return (ids, str_tokens) if return_str_tokens else ids 