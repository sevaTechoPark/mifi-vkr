"""Пакет cosine_similarity_classification: классификация по косинусной близости.

Включает три семейства методов:
- centroid: предсказание по близости к робастному центроиду класса;
- nearest: top-k soft-vote по ближайшим соседям;
- centroid_nn: взвешенная смесь centroid и nearest.
"""

from .main import run_from_params

__all__ = ["run_from_params"]
