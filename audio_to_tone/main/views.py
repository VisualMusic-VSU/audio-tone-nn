import os
import uuid
from collections import Counter

import librosa
import numpy as np
import tensorflow_hub as hub
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.metrics.pairwise import cosine_similarity

from .loader import YAMNET_MODEL, NLP_MODEL, CLASS_MAP, GENRE_MAP, MOOD_MAP
from .serializers import AudioFileSerializer


def load_yamnet_model():
    print("Загружаю модель YAMNet...")
    return hub.load("https://tfhub.dev/google/yamnet/1")


def predict_audio(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform = waveform.astype(np.float32)
    scores, _, _ = YAMNET_MODEL(waveform)
    return np.mean(scores, axis=0)


def interpret_genre(scores, max_genres):
    # Получаем топ 5 классов
    top_indices = np.argsort(scores)[-5:][::-1]
    top_classes = [CLASS_MAP[i] for i in top_indices]

    # Список базовых жанров (ключей в genre_map)
    base_genres = list(GENRE_MAP.keys())
    base_genre_embeddings = NLP_MODEL.encode(base_genres)

    best_base_genres = []
    for cls in top_classes:
        emb = NLP_MODEL.encode([cls])
        similarity = cosine_similarity(emb, base_genre_embeddings)[0]
        best_base_genres.append(base_genres[np.argmax(similarity)])
    best_base_genres = list(set(best_base_genres))  # уникальные базовые жанры

    # Уточнение поджанров с учетом похожести
    scored_subgenres = []
    for base_genre in best_base_genres:
        subgenre_names = list(GENRE_MAP[base_genre]['subgenres'].keys())
        subgenre_descriptions = list(GENRE_MAP[base_genre]['subgenres'].values())
        subgenre_texts = [desc[0] for desc in subgenre_descriptions]
        subgenre_embeddings = NLP_MODEL.encode(subgenre_texts)

        for cls in top_classes:
            emb = NLP_MODEL.encode([cls])
            similarity = cosine_similarity(emb, subgenre_embeddings)[0]

            for i, score in enumerate(similarity):
                scored_subgenres.append((subgenre_names[i], score))

    # Сортируем поджанры по убыванию похожести
    scored_subgenres.sort(key=lambda x: x[1], reverse=True)

    # Добавляем только уникальные названия
    best_subgenres = []
    seen = set()
    for name, _ in scored_subgenres:
        if name not in seen:
            best_subgenres.append(name)
            seen.add(name)
        if len(best_subgenres) == max_genres:
            break

    return best_subgenres


def interpret_mood(scores):
    # Получаем индексы топ-5 классов с наибольшими значениями scores
    top_indices = np.argsort(scores)[-5:][::-1]
    top_classes = [CLASS_MAP[i] for i in top_indices]

    # Получаем список ключевых слов для настроений из mood_map (словарь ключ: настроение)
    mood_keywords = list(MOOD_MAP.keys())
    mood_embeddings = NLP_MODEL.encode(mood_keywords)

    # Сопоставляем настроение
    detected = []
    for cls in top_classes:
        cls_emb = NLP_MODEL.encode([cls])
        similarity = cosine_similarity(cls_emb, mood_embeddings)[0]
        best_match_index = np.argmax(similarity)
        detected.append(MOOD_MAP[mood_keywords[best_match_index]])

    mood = Counter(detected).most_common(1)[0][0] if detected else "Indefinite"
    return mood


class AudioAnalysisAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = AudioFileSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        file = serializer.validated_data['file']
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join("temp", filename)

        try:
            os.makedirs("temp", exist_ok=True)
            with open(filepath, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            scores = predict_audio(filepath)
            genres = interpret_genre(scores, 5)
            mood = interpret_mood(scores)

            response_data = {
                "mood": mood,
                "genres": genres,
            }

            return Response(response_data)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
