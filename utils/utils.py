import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.harassment import harassment_template
from data.non_harassment import non_harassment_template, non_harassment_texts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def generate_all_templates(file_name):
    words_file = os.path.join(BASE_DIR, "data", file_name)
    words = load_words(words_file)

    all_texts = []
    for word in words:
        if file_name == "good_words.txt":
            all_texts.extend(non_harassment_template(word))
        else:
            all_texts.extend(harassment_template(word))

    return all_texts

def generate_harassments():
    return generate_all_templates("bad_words.txt")

def generate_non_harassments():
    texts = generate_all_templates("good_words.txt")
    texts.extend(non_harassment_texts)  # make sure this is a list
    return texts

# print(generate_harassments())