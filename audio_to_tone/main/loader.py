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


YAMNET_MODEL = tensorflow.saved_model.load(os.path.join(os.path.dirname(__file__), '../data/yamnet_model'))
NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

CLASS_MAP = load_class_map()
GENRE_MAP = load_json('../data/genre_map.json')
MOOD_MAP = load_json('../data/mood_map.json')

GENRE_MAP_CONVERTER = {
    "Alternative Rock": 1,
    "Hard Rock": 2,
    "Punk Rock": 3,
    "Indie Rock": 4,
    "Classic Rock": 5,

    "Electropop": 6,
    "Synthpop": 7,
    "Dance Pop": 8,
    "Indie Pop": 9,
    "Teen Pop": 10,

    "Trap": 11,
    "Boom Bap": 12,
    "Lo-fi": 13,
    "Gangsta Rap": 14,
    "Conscious Hip Hop": 15,

    "House": 16,
    "Techno": 17,
    "Trance": 18,
    "Drum and Bass": 19,
    "Dubstep": 20,

    "Bebop": 21,
    "Cool Jazz": 22,
    "Swing": 23,
    "Free Jazz": 24,
    "Fusion": 25,

    "Baroque": 26,
    "Romantic": 27,
    "Modern Classical": 28,
    "Renaissance": 29,
    "Minimalism": 30,

    "Black Metal": 31,
    "Death Metal": 32,
    "Nu Metal": 33,
    "Thrash Metal": 34,
    "Power Metal": 35,

    "Dub": 36,
    "Dancehall": 37,
    "Roots Reggae": 38,
    "Ska": 39,
    "Rocksteady": 40,

    "Delta Blues": 41,
    "Electric Blues": 42,
    "Chicago Blues": 43,
    "Country Blues": 44,
    "Soul Blues": 45,

    "Indietronica": 46,
    "Indie Folk": 47,

    "Afrobeat": 48,
    "Balkan": 49,
    "Cumbia": 50,
    "Flamenco": 51,
    "Klezmer": 52,

    "Ambient": 53,
    "Noise": 54,
    "Glitch": 55,
    "Drone": 56,
    "Avant-garde": 57
}

MOOD_MAP_CONVERTER = {
    "Cheerful": 1,
    "Neutral": 2,
    "Sad": 3,
    "Furious": 4,
    "Introspective": 5,
    "Assertive": 6,
    "Brutal": 7,
    "Energetic": 8,
    "Calm": 9,
    "Carefree": 10,
    "Confident": 11,
    "Danceable": 12,
    "Complex": 13,
    "Majestic":  14
}
