import re
import random
from typing import List, Tuple

def extract_labels(sentence: str):
    pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
    tokens = pattern.findall(sentence)
    words, init_labels, final_labels, cap_labels = [], [], [], []
    
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token, flags=re.UNICODE):

            # initial punctuation
            init = '¿' if i>0 and tokens[i-1]=='¿' else ''

            # final punctuation
            final = tokens[i+1] if i < len(tokens)-1 and tokens[i+1] in {'.',',','?'} else ''

            # capitalization
            if token.isupper():
                cap = 3
            elif token[0].isupper() and token[1:].islower():
                cap = 1
            elif token.islower():
                cap = 0
            else:
                cap = 2

            words.append(token)
            init_labels.append(init)
            final_labels.append(final)
            cap_labels.append(cap)

    return words, init_labels, final_labels, cap_labels

def split_data_from_file(
    filepath: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    max_lines: int = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Loads sentences from a .txt file (one sentence per line),
    shuffles them, and splits into train/val/test lists.

    Returns (train_data, val_data, test_data) as lists of strings.
    """

    # 1. Load lines
    with open(filepath, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. Optionally limit
    if max_lines is not None:
        lines = lines[:max_lines]

    # 3. Check ratios
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    # 4. Shuffle
    random.shuffle(lines)

    # 5. Split
    n = len(lines)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]

    return train_data, val_data, test_data