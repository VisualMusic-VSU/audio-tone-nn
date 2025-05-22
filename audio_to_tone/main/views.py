import numpy as np
import tensorflow_hub as hub
import librosa
import urllib.request
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import AudioFileSerializer

GENRE_MAP = {
    "Hip hop music": ["Рэп", "Трэп", "Бум-бэп", "Дрилл"],
    "Rock music": ["Хард-рок", "Альтернативный рок", "Метал"],
    "Electronic music": ["Хаус", "Техно", "Драм-н-бейс"],
    "Jazz": ["Смуз-джаз", "Бибоп", "Фьюжн"],
    "Classical music": ["Симфония", "Оркестровая музыка", "Камерная музыка"]
}

MOOD_MAP = {
    "laughter": "Весёлое",
    "speech": "Нейтральное",
    "crying": "Грустное",
    "screaming": "Яростное",
    "whispering": "Интроспективное",
    "aggressive": "Напористое",
    "heavy metal": "Брутальное",
    "rock music": "Энергичное",
    "relaxing": "Успокаивающее",
    "pop music": "Беззаботное",
    "hip hop music": "Уверенное",
    "electronic music": "Танцевальное",
    "jazz": "Замысловатое",
    "classical music": "Величественное"
}


def load_yamnet_model():
    print("Загружаю модель YAMNet...")
    return hub.load("https://tfhub.dev/google/yamnet/1")


def load_class_map():
    url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    class_map = {}
    response = urllib.request.urlopen(url)
    lines = [line.decode('utf-8') for line in response.readlines()]
    reader = csv.reader(lines)
    next(reader)
    for row in reader:
        if len(row) >= 3:
            class_map[int(row[0])] = row[2]
    return class_map


def predict_audio(audio_path, model):
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform = waveform.astype(np.float32)
    scores, _, _ = model(waveform)
    return np.mean(scores, axis=0)


def interpret_genre(scores, class_map, nlp_model):
    top_indices = np.argsort(scores)[-5:][::-1]
    top_classes = [class_map[i] for i in top_indices]

    genre_descriptions = list(GENRE_MAP.keys())
    genre_embeddings = nlp_model.encode(genre_descriptions)

    best_genres = []
    for cls in top_classes:
        emb = nlp_model.encode([cls])
        similarity = cosine_similarity(emb, genre_embeddings)[0]
        best_genres.append(genre_descriptions[np.argmax(similarity)])

    best_genres = list(set(best_genres))

    related = []
    for genre in best_genres:
        related.extend(GENRE_MAP.get(genre, []))

    return best_genres, list(set(related)), top_classes


def interpret_mood(scores, class_map):
    top_indices = np.argsort(scores)[-5:][::-1]
    top_classes = [class_map[i] for i in top_indices]

    detected = []
    for cls in top_classes:
        for keyword, label in MOOD_MAP.items():
            if keyword.lower() in cls.lower():
                detected.append(label)
                break

    if detected:
        mood = max(set(detected), key=detected.count)
    else:
        mood = "Неопределённое"

    return mood, top_classes


class AudioAnalysisAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = AudioFileSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']

            with open("temp_audio.wav", "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            yamnet_model = load_yamnet_model()
            class_map = load_class_map()
            nlp_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

            scores = predict_audio("temp_audio.wav", yamnet_model)
            main_genres, related_genres, top_classes = interpret_genre(scores, class_map, nlp_model)
            mood, _ = interpret_mood(scores, class_map)

            response_data = {
                "mood": mood,
                "genres": top_classes
            }
            return Response(response_data)
        return Response(serializer.errors, status=400)
