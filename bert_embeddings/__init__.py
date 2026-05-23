"""Пакет bert_embeddings: дообучение и инференс энкодера ruRoberta для длинных текстов."""

from .embedding_model import LongTextRobertaEmbedder

__all__ = ["LongTextRobertaEmbedder"]
