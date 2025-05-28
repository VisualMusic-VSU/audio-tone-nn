import csv
import json
import os

import tensorflow
from sentence_transformers import SentenceTransformer


def load_class_map():
    path = os.path.join(os.path.dirname(__file__), '../data/yamnet_class_map.csv')
    class_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                class_map[int(row[0])] = row[2]
    return class_map


def load_json(path):
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path, "r", encoding="utf-8") as file:
        genre_map_extended = json.load(file)
    return genre_map_extended


YAMNET_MODEL = tensorflow.saved_model.load('./data/yamnet_model')
NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

CLASS_MAP = load_class_map()
GENRE_MAP = load_json('../data/genre_map.json')
MOOD_MAP = load_json('../data/mood_map.json')
