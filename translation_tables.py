#!/usr/bin/env python3
"""
Translation table generation and perplexity comparison for IBM Model 1
"""

import math
import random
import pickle
from collections import Counter
from typing import List, Tuple

from config import (
    TOP_N_SOURCE_WORDS, TOP_N_TRANSLATIONS, 
    PREPROCESSED_DATA_FILE, FINAL_MODEL_FILE
)
from training import IBMModel1


def load_preprocessed_data(filepath: str) -> List[Tuple[List[str], List[str]]]:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_top_translations(model: IBMModel1, source_word: str, n: int = 5) -> List[Tuple[str, float]]:
    translations = []
    for e in model.target_vocab:
        prob = model.t[e][source_word]
        if prob > 0:
            translations.append((e, prob))
    translations.sort(key=lambda x: x[1], reverse=True)
    return translations[:n]


def sentence_log2_probability(model: IBMModel1, source_sent: List[str], target_sent: List[str]) -> float:
    """
    log₂ P(e|f) = Σⱼ log₂( (1/(l+1)) × Σᵢ t(eⱼ|fᵢ) )
    """
    source_with_null = ["NULL"] + source_sent
    l_plus_1 = len(source_with_null)
    
    log2_prob = 0.0
    for e in target_sent:
        prob_e = sum(model.t[e][f] for f in source_with_null)
        prob_e = prob_e / l_plus_1
        
        if prob_e > 0:
            log2_prob += math.log2(prob_e)
        else:
            log2_prob += math.log2(1e-10)
    
    return log2_prob


def log2_perplexity(model: IBMModel1, source_sent: List[str], target_sent: List[str]) -> float:
    """
    log₂ PP = -Σ log₂ p(e|f)
    """
    log2_prob = sentence_log2_probability(model, source_sent, target_sent)
    if len(target_sent) == 0:
        return float('inf')
    return -log2_prob


def perplexity(model: IBMModel1, source_sent: List[str], target_sent: List[str]) -> float:
    """
    PP = 2^(-log₂_prob)
    """
    log2_pp = log2_perplexity(model, source_sent, target_sent)
    if log2_pp == float('inf'):
        return float('inf')
    if log2_pp > 1000:
        return float('inf')
    return math.pow(2, log2_pp)


def print_translation_tables(model: IBMModel1, parallel_data: List[Tuple[List[str], List[str]]]):
    print(f"\nTranslation Tables: Top {TOP_N_TRANSLATIONS} translations for {TOP_N_SOURCE_WORDS} most common Spanish words\n")
    
    source_freq = Counter()
    for f_sent, e_sent in parallel_data:
        source_freq.update(f_sent)
    
    most_common = source_freq.most_common(TOP_N_SOURCE_WORDS)
    
    for source_word, freq in most_common:
        print(f"'{source_word}' (freq: {freq})")
        translations = get_top_translations(model, source_word, TOP_N_TRANSLATIONS)
        for rank, (target_word, prob) in enumerate(translations, 1):
            print(f"  {rank}. '{target_word}': {prob:.6f}")
        print()


def compare_perplexity(model: IBMModel1, parallel_data: List[Tuple[List[str], List[str]]]):
    print("Perplexity Comparison: Real vs Random Translations\n")
    
    target_vocab_list = list(model.target_vocab)
    sample_indices = random.sample(range(len(parallel_data)), 5)
    
    for i, idx in enumerate(sample_indices, 1):
        f_sent, e_sent_real = parallel_data[idx]
        e_sent_random = random.choices(target_vocab_list, k=len(e_sent_real))
        
        log2_ppl_real = log2_perplexity(model, f_sent, e_sent_real)
        log2_ppl_random = log2_perplexity(model, f_sent, e_sent_random)
        
        print(f"Example {i}:")
        print(f"  Source (Spanish):  {' '.join(f_sent[:12])}{'...' if len(f_sent) > 12 else ''}")
        print(f"  Real translation:  {' '.join(e_sent_real[:12])}{'...' if len(e_sent_real) > 12 else ''}")
        print(f"  Random sentence:   {' '.join(e_sent_random[:12])}{'...' if len(e_sent_random) > 12 else ''}")
        print(f"  log₂ PP (real):   {log2_ppl_real:,.2f}")
        print(f"  log₂ PP (random): {log2_ppl_random:,.2f}")
        
        if log2_ppl_real < log2_ppl_random:
            print(f"  Result: Real translation has lower perplexity (diff: {log2_ppl_random - log2_ppl_real:,.2f} bits)")
        else:
            print(f"  Result: Unexpected - random has lower perplexity")
        print()


def main():
    random.seed(42)
    
    print(f"Loading data from {PREPROCESSED_DATA_FILE}...")
    parallel_data = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    print(f"Loaded {len(parallel_data)} sentence pairs.\n")
    
    print(f"Loading model from {FINAL_MODEL_FILE}...")
    model = IBMModel1.load(FINAL_MODEL_FILE)
    print(f"Model loaded ({model.current_iteration} iterations)\n")
    
    print_translation_tables(model, parallel_data)
    compare_perplexity(model, parallel_data)


if __name__ == "__main__":
    main()
