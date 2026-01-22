#!/usr/bin/env python3
"""Configuration for IBM Model 1"""

NUM_TRAINING_PAIRS = 10000
NUM_EM_ITERATIONS = 5
TOP_N_SOURCE_WORDS = 10
TOP_N_TRANSLATIONS = 5

DATA_DIR = "europarl_data"
CHECKPOINT_DIR = "checkpoints"
ES_FILE = f"{DATA_DIR}/europarl-v7.es-en.es"
EN_FILE = f"{DATA_DIR}/europarl-v7.es-en.en"
PREPROCESSED_DATA_FILE = "preprocessed_data.pkl"
MODEL_CHECKPOINT_PREFIX = f"{CHECKPOINT_DIR}/model_iter"
FINAL_MODEL_FILE = f"{CHECKPOINT_DIR}/model_final.pkl"
