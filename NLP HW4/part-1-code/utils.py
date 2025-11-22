import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    text = example["text"]

    # Hyperparameters for transformation (tunable knobs)
    # Increase carefully; too aggressive can become unreasonable.
    p_synonym = 0.25   # probability to replace a token with a synonym
    p_typo = 0.15      # probability to introduce a single-character typo
    p_lowercase = 0.8  # probability to lowercase whole sentence
    p_delete = 0.05    # probability to delete a non-protected medium-length token
    p_punct_strip = 0.3  # probability to strip trailing punctuation from a token
    p_filler = 0.15      # probability to prepend/append a neutral filler phrase

    # Tokens to avoid changing (sentiment-bearing words, simple heuristic)
    protected_words = set([
        "good", "great", "excellent", "amazing", "fantastic", "wonderful", "love", "liked", "favorite",
        "bad", "terrible", "awful", "horrible", "worst", "hate", "disliked", "boring", "dull",
        # extra protections to avoid sentiment flips
        "not", "no", "never", "poor", "mediocre", "enjoy", "enjoyed", "enjoyable",
        "recommend", "recommended", "recommendation", "must", "must-see", "masterpiece"
    ])

    # Occasionally lowercase the entire sentence to stress test a cased model
    if random.random() < p_lowercase:
        text = text.lower()

    # Optionally add neutral filler phrases (keeps sentiment label intact)
    filler_phrases_pre = ["Honestly,", "Frankly,", "In my opinion,", "To be fair,", "From my view,"]
    filler_phrases_post = ["overall.", "to be honest.", "in the end.", "if you ask me."]
    if random.random() < p_filler:
        prefix = random.choice(filler_phrases_pre)
        text = prefix + " " + text
    if random.random() < p_filler:
        suffix = random.choice(filler_phrases_post)
        text = text.rstrip() + " " + suffix

    tokens = word_tokenize(text)
    detok = TreebankWordDetokenizer()

    # QWERTY neighbor mapping for a subset of letters for realism
    neighbor_map = {
        'a': ['s', 'q', 'w', 'z'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 's', 'd', 'r'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'k', 'l', 'p'], 'p': ['o', 'l'],
        'q': ['w', 'a'], 'r': ['e', 'd', 'f', 't'], 's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'f', 'g', 'y'], 'u': ['y', 'h', 'j', 'i'], 'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'], 'x': ['z', 's', 'd', 'c'], 'y': ['t', 'g', 'h', 'u'], 'z': ['x', 's', 'a']
    }

    def choose_synonym(word: str) -> str:
        syns = wordnet.synsets(word)
        if not syns:
            return word
        lemmas = set()
        for s in syns:
            for l in s.lemmas():
                cand = l.name().replace('_', ' ').strip()
                if cand.lower() != word.lower() and cand.isalpha():
                    lemmas.add(cand)
        if not lemmas:
            return word
        replacement = random.choice(list(lemmas))
        # Preserve capitalization if original is capitalized
        if word[0].isupper():
            replacement = replacement.capitalize()
        return replacement

    def inject_typo(word: str) -> str:
        # apply a single-character substitution with a neighboring key
        if len(word) == 0:
            return word
        # pick a position that is alphabetic
        indices = [i for i, ch in enumerate(word) if ch.isalpha()]
        if not indices:
            return word
        pos = random.choice(indices)
        ch = word[pos]
        lower = ch.lower()
        if lower not in neighbor_map:
            return word
        repl = random.choice(neighbor_map[lower])
        # preserve case
        repl = repl.upper() if ch.isupper() else repl
        return word[:pos] + repl + word[pos+1:]

    new_tokens = []
    for tok in tokens:
        # Handle deletion (skip very short tokens & protected)
        if tok.isalpha() and tok.lower() not in protected_words and 4 <= len(tok) <= 10:
            if random.random() < p_delete:
                continue  # drop token

        base_tok = tok
        applied = False
        if tok.isalpha() and tok.lower() not in protected_words:
            if random.random() < p_synonym:
                syn = choose_synonym(tok)
                if syn != tok:
                    base_tok = syn
                    applied = True
        if tok.isalpha() and not applied and random.random() < p_typo:
            base_tok = inject_typo(base_tok)

        # Optional punctuation stripping (e.g., remove trailing period/comma)
        if random.random() < p_punct_strip and len(base_tok) > 2 and not base_tok.isalpha():
            if base_tok[-1] in [',', '.', '!', '?', ';', ':']:
                base_tok = base_tok[:-1]

        new_tokens.append(base_tok)

    example["text"] = detok.detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
