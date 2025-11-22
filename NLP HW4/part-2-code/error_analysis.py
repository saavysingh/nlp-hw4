"""Lightweight error analysis for text-to-SQL predictions.

Given:
  gold SQL file (one query per line)
  predicted SQL file (one query per line)
  NL file (one question per line) to enrich examples

Outputs:
  JSON summary with category counts and representative examples.

Usage (from repo root):
  python3 hw4-code/part-2-code/error_analysis.py \
      --gold data/dev.sql \
      --nl data/dev.nl \
      --pred results/t5_ft_experiment_dev.sql \
      --out results/error_analysis_ft.json

Categories (heuristic, non-exhaustive):
  EXACT_MATCH        – prediction identical to gold
  SYNTAX_ERROR       – obvious malformed tokens or paren imbalance
  STRUCTURE_ERROR    – SELECT target mismatch or aggregate mismatch
  MISSING_CONDITION  – gold has condition tokens absent in prediction
  EXTRA_CONDITION    – prediction adds spurious tokens/conditions
  VALUE_MISMATCH     – literals differ (numbers / quoted strings)
  JOIN_MISMATCH      – missing table alias present in gold FROM/JOIN

Heuristics favor recall of distinct error classes rather than perfect precision.
They are designed for quick qualitative analysis for the report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Example:
    idx: int
    nl: str
    gold: str
    pred: str
    categories: List[str]


PAREN_RE = re.compile(r'[()]')
STRING_LITERAL_RE = re.compile(r"'([^']*)'")
NUM_LITERAL_RE = re.compile(r"\b\d+\b")
SELECT_TARGET_RE = re.compile(r"^\s*SELECT\s+DISTINCT\s+(.+?)\s+FROM\s", re.IGNORECASE)
SELECT_ANY_RE = re.compile(r"^\s*SELECT\s+(.+?)\s+FROM\s", re.IGNORECASE)
TABLE_ALIAS_RE = re.compile(r"\b([a-z_]+_[0-9]+)\b")


def load_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def tokenize(sql: str) -> List[str]:
    # crude tokenization: split on spaces and punctuation boundaries
    spaced = re.sub(r"([(),=<>])", r" \1 ", sql)
    spaced = re.sub(r"\s+", " ", spaced)
    return spaced.strip().split()


def paren_balance(sql: str) -> int:
    return sql.count('(') - sql.count(')')


def extract_select_target(sql: str) -> str:
    m = SELECT_TARGET_RE.search(sql) or SELECT_ANY_RE.search(sql)
    if not m:
        return ''
    return m.group(1).strip().lower()


def extract_literals(sql: str) -> Tuple[List[str], List[str]]:
    strings = STRING_LITERAL_RE.findall(sql)
    nums = NUM_LITERAL_RE.findall(sql)
    return strings, nums


def extract_aliases(sql: str) -> List[str]:
    # FROM/JOIN area heuristic: everything after FROM up to WHERE
    if ' from ' not in sql.lower():
        return []
    lower = sql.lower()
    from_idx = lower.find(' from ')
    where_idx = lower.find(' where ')
    segment = lower[from_idx: where_idx if where_idx != -1 else len(sql)]
    return sorted(set(TABLE_ALIAS_RE.findall(segment)))


def conditions_set(sql: str) -> set:
    # Split on AND / OR boundaries to approximate individual conditions
    tmp = re.split(r"\bAND\b|\bOR\b", sql, flags=re.IGNORECASE)
    conds = set()
    for c in tmp:
        c = c.strip()
        if not c:
            continue
        # ignore trivial ALWAYS TRUE fragments
        if c in {'1 = 1'}:
            continue
        # shrink whitespace
        c_norm = re.sub(r"\s+", " ", c)
        # keep moderately sized conditions
        if 3 <= len(c_norm.split()) <= 25:
            conds.add(c_norm.lower())
    return conds


def classify(gold: str, pred: str) -> List[str]:
    cats: List[str] = []
    if gold == pred:
        return ['EXACT_MATCH']

    # Syntax issues
    if paren_balance(pred) != 0 or re.search(r"\bnot\s*\(\s*\(\s*", pred.lower()) or re.search(r"\b=\s*\)", pred):
        cats.append('SYNTAX_ERROR')
    if re.search(r"\barrival_time\s{2,}\d", pred):
        cats.append('SYNTAX_ERROR')

    # Structure
    gold_target = extract_select_target(gold)
    pred_target = extract_select_target(pred)
    if gold_target and pred_target and gold_target != pred_target:
        cats.append('STRUCTURE_ERROR')

    # Aliases (joins)
    g_alias = extract_aliases(gold)
    p_alias = extract_aliases(pred)
    missing_alias = set(g_alias) - set(p_alias)
    if missing_alias:
        cats.append('JOIN_MISMATCH')

    # Literals
    g_strings, g_nums = extract_literals(gold)
    p_strings, p_nums = extract_literals(pred)
    if (set(g_strings) - set(p_strings)) or (set(g_nums) - set(p_nums)):
        cats.append('VALUE_MISMATCH')

    # Conditions
    g_conds = conditions_set(gold)
    p_conds = conditions_set(pred)
    missing = g_conds - p_conds
    extra = p_conds - g_conds
    if missing:
        cats.append('MISSING_CONDITION')
    if extra:
        cats.append('EXTRA_CONDITION')

    # If no category triggered, fall back to STRUCTURE_ERROR to flag difference
    if not cats:
        cats.append('STRUCTURE_ERROR')
    return cats


def analyze(gold_lines: List[str], pred_lines: List[str], nl_lines: List[str]) -> Dict:
    total = min(len(gold_lines), len(pred_lines), len(nl_lines))
    stats = Counter()
    examples_by_cat: Dict[str, List[Example]] = defaultdict(list)
    for i in range(total):
        gold = gold_lines[i]
        pred = pred_lines[i]
        nl = nl_lines[i]
        cats = classify(gold, pred)
        for c in cats:
            stats[c] += 1
            if len(examples_by_cat[c]) < 5:  # keep first 5
                examples_by_cat[c].append(Example(i, nl, gold, pred, cats))

    accuracy = stats['EXACT_MATCH'] / total if total else 0.0
    return {
        'total_pairs': total,
        'category_counts': stats,
        'exact_match_ratio': accuracy,
        'examples': {
            cat: [e.__dict__ for e in lst] for cat, lst in examples_by_cat.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True, help='Gold SQL file path')
    ap.add_argument('--pred', required=True, help='Predicted SQL file path')
    ap.add_argument('--nl', required=True, help='Natural language questions file path')
    ap.add_argument('--out', required=True, help='Output JSON path')
    args = ap.parse_args()

    gold_lines = load_lines(args.gold)
    pred_lines = load_lines(args.pred)
    nl_lines = load_lines(args.nl)

    result = analyze(gold_lines, pred_lines, nl_lines)
    # Convert Counter to dict for JSON
    result['category_counts'] = dict(result['category_counts'])
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote analysis to {args.out}. Exact match: {result['exact_match_ratio']:.3%}")


if __name__ == '__main__':
    main()
