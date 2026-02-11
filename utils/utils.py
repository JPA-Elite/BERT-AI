import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.harassment import harassment_texts
from data.harassment_tagalog import harassment_texts_tagalog
from data.non_harassment import non_harassment_texts
from data.non_harassment_tagalog import non_harassment_texts_tagalog


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def generate_harassments():
    return harassment_texts + harassment_texts_tagalog

def generate_non_harassments():
    return non_harassment_texts + non_harassment_texts_tagalog
