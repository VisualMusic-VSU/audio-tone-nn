import librosa
import numpy as np
import os
import tensorflow_hub as hub
import uuid
from collections import Counter
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.metrics.pairwise import cosine_similarity

from .loader import YAMNET_MODEL, NLP_MODEL, CLASS_MAP, GENRE_MAP, MOOD_MAP
from .serializers import AudioFileSerializer


def predict_audio(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
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
    # Получаем топ-5 индексов и классов с наибольшими score
    top_indices = np.argsort(scores)[-5:][::-1]
    top_classes = [CLASS_MAP[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    # Разбираем ключи MOOD_MAP на отдельные синонимы и готовим списки
    mood_phrases = []
    mood_labels = []
    for key, mood in MOOD_MAP.items():
        synonyms = [phrase.strip() for phrase in key.split(",")]
        mood_phrases.extend(synonyms)
        mood_labels.extend([mood] * len(synonyms))

    # Кодируем все фразы настроений
    mood_embeddings = NLP_MODEL.encode(mood_phrases, convert_to_tensor=False)

    detected = []
    for cls, score in zip(top_classes, top_scores):
        # Разбиваем класс (если содержит синонимы) и усредняем эмбеддинги
        cls_synonyms = [s.strip() for s in cls.split(",")]
        cls_embs = NLP_MODEL.encode(cls_synonyms, convert_to_tensor=False)
        cls_emb = np.mean(cls_embs, axis=0, keepdims=True)

        # Считаем косинусное сходство с каждым из mood_phrases
        similarity = cosine_similarity(cls_emb, mood_embeddings)[0]
        best_idx = np.argmax(similarity)

        if similarity[best_idx] >= similarity_threshold:
            matched_mood = mood_labels[best_idx]
            # Взвешиваем по степени уверенности модели (score * similarity)
            detected.append((matched_mood, similarity[best_idx] * score))

    if detected:
        # Суммируем веса для каждого настроения
        mood_scores = {}
        for mood, weight in detected:
            mood_scores[mood] = mood_scores.get(mood, 0) + weight

        # Возвращаем настроение с максимальным суммарным весом
        return max(mood_scores.items(), key=lambda x: x[1])[0]
    else:
        # Если совпадений нет — считаем настроение нейтральным
        return "Neutral"


def genres_to_ids(genres_list):
    return [GENRE_MAP_CONVERTER.get(genre, -1) for genre in genres_list]


def mood_to_id(mood_str):
    return MOOD_MAP_CONVERTER.get(mood_str, -1)


class AudioAnalysisAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        operation_description="Анализирует аудио файл и возвращает настроение и жанры",
        request_body=AudioFileSerializer,
        responses={
            200: openapi.Response(
                description="Анализ завершен успешно",
            ),
            400: openapi.Response(
                description="Ошибки в запросе",
            ),
        }
    )
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
                "mood": mood_to_id(mood),
                "genres": genres_to_ids(genres),
            }

            return Response(response_data)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
