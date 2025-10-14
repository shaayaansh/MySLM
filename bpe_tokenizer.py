import os
from pathlib import Path
import regex as re
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import parallelize_tokenize_file
from typing import List, Tuple, Iterable, Iterator, Optional, Union
from collections import Counter
import unicodedata as ud


class BytePairEncodingTokenizer():
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        super().__init__()
        
        self.merges = []
        self.merge_ranks = {}
        self.b2u, self.u2b = self._bytes_to_unicode()
        self.token_to_id = {}
        self.id_to_token = []

        self.special_tokens = list(special_tokens) if special_tokens else []
        self.special_tokens_set = set(self.special_tokens)

        self.DEFAULT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._word_re = re.compile(self.DEFAULT_PAT)

        if vocab is not None:
            self._load_vocab(vocab)
        if merges is not None:
            self._load_merges(merges)
  
    def _load_vocab(self, vocab):
        if isinstance(vocab, (str, Path)):
            with Path(vocab).open("r", encoding="utf-8") as f:
                vocab = json.load(f)
        
        if isinstance(vocab, list):
            self.id_to_token = [str(t) for t in vocab]
            self.token_to_id = {t: i for i, t in enumerate(vocab)}
        
        elif isinstance(vocab, dict):
            tok2id = {str(k): int(v) for k, v in vocab.items()}
            size = 1 + max(tok2id.values()) if tok2id else 0
            id2tok = [None] * size

            for tok, idx in tok2id.items():
                if 0 <= idx < size and id2tok[idx] is None:
                    id2tok[idx] = tok
                else:
                    raise ValueError("vocab has non-contiguous or duplicate ids.")
            
            if any(t is None for t in id2tok):
                missing = [i for i, t in enumerate(id2tok) if t is None][:20]
                raise ValueError(f"vocab has gaps in ids; example missing: {missing}")
            
            self.token_to_id = tok2id
            self.id_to_token = id2tok
        
        else:
            raise TypeError("vocab must be a path, list[str], or dict[str,int].")
        
    def _load_merges(self, merges):
        if isinstance(merges, (str, Path)):
            pairs: List[Tuple[str, str]] = []
            with Path(merges).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    a, b = line.split(" ", 1)
                    pairs.append((a, b))
        else:
            pairs = list(merges)
        
        self.merges = pairs
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        self = cls(special_tokens=special_tokens)
        self._load_vocab(vocab_path)
        self._load_merges(merges_path)

        return self

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


    def initialize_vocabulary(self, special_tokens):
        self.token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token = special_tokens[:]
        
        # base byte tokens (each is a single printable Unicode "byte-char")
        for b in range(256):
            tok = bytes([b])
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

        if not all_token_counts:
            return []
        
        # Sort by (-frequency, pair) for deterministic GPT-2 behavior
        best_pair = sorted(all_token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return [best_pair]
    
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
            tuple(bytes([bb]) for bb in s.encode("utf-8")): freq
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
        
        return {i: tok for i, tok in enumerate(self.id_to_token)}, self.merges
    
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
    
    def _tokens_to_ids_iter(self, tokens: Iterable[str], strict: bool = False) -> Iterator[int]:
        for tok in tokens:
            tid = self.token_to_id.get(tok)
            if tid is not None:
                yield tid
            elif strict:
                raise KeyError(f"Unknown token: {tok!r}")
            else:
                for ch in tok:
                    base_id = self.token_to_id.get(ch)
                    if base_id is None:
                        raise KeyError(f"Missing base byte token for {ch!r}")
                    yield base_id
    

    def encode(self, text: str, return_str_tokens: bool = False):
        if return_str_tokens:
            str_tokens = []
            for w in self._word_re.findall(text):
                str_tokens.extend(self._encode_word(w))

            ids = list(self._tokens_to_ids_iter(str_tokens, strict=False))
            return ids, str_tokens
        
        ids_iter = (
            tid
            for w in self._word_re.findall(text)
            for tid in self._tokens_to_ids_iter(self._encode_word(w), strict=False)
        )

        return list(ids_iter)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily yield token IDs from an iterable of strings.
        Memory-efficient for huge files.
        """
        for chunk in iterable:
            for w in self._word_re.findall(chunk):
                yield from self._tokens_to_ids_iter(self._encode_word(w), strict=False)
        

    def decode(self, ids: List[int]) -> str:
        byte_vals = []
        for i in ids:
            tok = self.id_to_token[i]
            for ch in tok:
                byte_vals.append(self.u2b[ch])

        return bytes(byte_vals).decode("utf-8", errors="strict")
    


def bytes_to_unicode():
    """
    Returns a dictionary mapping bytes (0–255) to unique printable Unicode strings,
    following GPT-2’s original byte-to-unicode scheme.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # === Store byte ↔ unicode mapping ===
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        # === Build merge ranks (pair → rank index) ===
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # === Convert vocab bytes → printable unicode ===
        self.id_to_token = ["".join(self.b2u[b] for b in token) for _, token in sorted(vocab.items())]
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

        self._split_pat = re.compile(r" ?\S+|\s+")

        # === Handle special tokens ===
        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token.append(tok)

    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """Construct a Tokenizer from serialized vocab and merges."""
        import json

        # Load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_dict = json.load(vf)
            vocab = {int(k): bytes(v.encode("latin-1")) for k, v in vocab_dict.items()}

        # Load merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                if line.strip() == "":
                    continue
                a, b = line.rstrip().split(" ")
                merges.append((a.encode("latin-1"), b.encode("latin-1")))

        return cls(vocab, merges, special_tokens=special_tokens)

    # ------------------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        """Apply BPE merges to a single token (in unicode form)."""
        # Convert printable Unicode chars back to bytes for merge lookups
        word = tuple(bytes([self.u2b[c]]) for c in token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return [token]

        while True:
            ranked_pairs = [(self.merge_ranks.get(p, float("inf")), p) for p in pairs]
            best_rank, best_pair = min(ranked_pairs)
            if best_rank == float("inf"):
                break

            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}

        # Convert merged bytes back to printable unicode tokens
        return ["".join(self.b2u[b] for b in w) for w in word]

    def encode(self, text: str) -> list[int]:
        import regex as re

        specials = self.special_tokens or []
        specials_sorted = sorted(specials, key=len, reverse=True)
        pat_special = re.compile("|".join(re.escape(s) for s in specials_sorted)) if specials else None

        # --- Split on special tokens so that they form their own segments ---
        parts = []
        if pat_special:
            # Split text into regular segments and special tokens
            split_text = re.split(f"({pat_special.pattern})", text)
            for seg in split_text:
                if not seg:
                    continue
                if seg in specials:
                    parts.append((True, seg))
                else:
                    parts.append((False, seg))
        else:
            parts = [(False, text)]

        # --- Encode each segment independently ---
        out = []
        for i, (is_special, seg) in enumerate(parts):
            if is_special:
                tid = self.token_to_id.get(seg)
                if tid is not None:
                    out.append(tid)
            else:
                # Check if previous segment was a special token
                prev_was_special = i > 0 and parts[i - 1][0]
                
                if prev_was_special and seg:
                    # Check if segment matches pattern: MULTIPLE whitespace chars followed by non-whitespace
                    # Single space + text should be encoded normally (e.g., " are" -> token 389)
                    # Multiple whitespace + text should be split (e.g., "\n\nHello" -> [198, 198, ...])
                    if seg and len(seg) > 0:
                        # Find where non-whitespace starts
                        ws_end = 0
                        for ch in seg:
                            if ch.isspace():
                                ws_end += 1
                            else:
                                break
                        
                        # Only split if we have 2+ whitespace chars AND there's non-whitespace after
                        if ws_end >= 2 and ws_end < len(seg):
                            ws_part = seg[:ws_end]
                            rest_part = seg[ws_end:]
                            
                            # Encode whitespace part byte-by-byte
                            ws_bytes = ws_part.encode("utf-8")
                            ws_printable = "".join(self.b2u[bb] for bb in ws_bytes)
                            for ch in ws_printable:
                                cid = self.token_to_id.get(ch)
                                if cid is not None:
                                    out.append(cid)
                            
                            # Encode rest normally
                            out.extend(self._encode_regular(rest_part, no_merge=False))
                        else:
                            # No split needed (either single space, no leading ws, or only ws)
                            out.extend(self._encode_regular(seg, no_merge=False))
                    else:
                        out.extend(self._encode_regular(seg, no_merge=False))
                else:
                    out.extend(self._encode_regular(seg, no_merge=False))

        return out


    def _encode_regular(self, text: str, no_merge: bool = False) -> list[int]:
        b = text.encode("utf-8")
        printable = "".join(self.b2u[bb] for bb in b)
        tokens = self._split_pat.findall(printable)
        out_ids = []
        
        for tok in tokens:
            if no_merge:
                # Encode byte-by-byte without BPE merges
                for ch in tok:
                    cid = self.token_to_id.get(ch)
                    if cid is not None:
                        out_ids.append(cid)
            else:
                # Normal BPE encoding
                for sub in self._bpe(tok):
                    tid = self.token_to_id.get(sub)
                    if tid is not None:
                        out_ids.append(tid)
                    else:
                        for ch in sub:
                            cid = self.token_to_id.get(ch)
                            if cid is not None:
                                out_ids.append(cid)
        return out_ids
            
    # ------------------------------------------------------------------
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings."""
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to readable text."""
        # Convert back from printable unicode → bytes
        text = "".join(self.id_to_token[i] for i in ids)
        decoded_bytes = bytes([self.u2b[c] for c in text if c in self.u2b])
        return decoded_bytes.decode("utf-8", errors="replace")