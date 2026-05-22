"""Алгоритмы классификации по косинусной близости.

Три семейства методов:
- centroid: робастный центроид класса (hard-trim / soft-trim + iterative refinement);
- nearest: top-k soft-vote по соседям с температурой;
- centroid_nn: взвешенный ансамбль centroid + nearest.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-нормирует каждую строку матрицы (с epsilon-защитой от нулей)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


# =============================================================================
# Centroid: weighted / iterative
# =============================================================================

def _centroid_one_class_soft(
    class_embs: np.ndarray,
    trim_ratio: float,
    trim_power: float,
    refine_iters: int,
    mode: str,
) -> np.ndarray:
    """Робастный центроид одного класса.

    mode="hard":
        1) обычный mean → L2;
        2) cos-sim каждой точки к центроиду; отбрасываем нижние trim_ratio долей;
        3) mean оставшихся → L2.

    mode="soft":
        1) обычный mean → L2;
        2) cos-sim каждой точки к центроиду; вес = max(sim, 0) ** trim_power;
        3) опционально зануляем веса у нижних trim_ratio долей
           (устойчивость к мислейблам и выбросам);
        4) weighted mean → L2.

    refine_iters задаёт количество дополнительных итераций пересчёта центроида:
        0 — один проход;
        1 — ещё одна итерация поверх (обычно сходится за 1-2 шага).
    """
    n = class_embs.shape[0]
    if n <= 1:
        c = class_embs.mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-12)

    centroid = class_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    iters = max(1, 1 + int(refine_iters))

    for _ in range(iters):
        sims = class_embs @ centroid  # (n,)
        if mode == "hard":
            n_keep = max(1, int(np.ceil(n * (1.0 - trim_ratio))))
            keep_idx = np.argpartition(-sims, n_keep - 1)[:n_keep]
            kept = class_embs[keep_idx]
            new_centroid = kept.mean(axis=0)
        else:  # soft
            w = np.clip(sims, 0.0, None) ** float(trim_power)
            if trim_ratio > 0.0 and n >= 3:
                # Зануляем веса нижних trim_ratio*n точек по cos-sim.
                n_zero = int(np.floor(n * trim_ratio))
                if n_zero > 0:
                    cutoff = np.partition(sims, n_zero)[n_zero - 1]
                    w[sims <= cutoff] = 0.0
            wsum = w.sum() + 1e-12
            new_centroid = (class_embs * w[:, None]).sum(axis=0) / wsum

        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-12)

        # Ранний выход, если центроид перестал сдвигаться.
        if np.linalg.norm(new_centroid - centroid) < 1e-6:
            centroid = new_centroid
            break
        centroid = new_centroid

    return centroid


def build_centroids(
    train_embs: np.ndarray,
    train_labels,
    trim_ratio: float = 0.0,
    trim_mode: str = "hard",
    trim_power: float = 4.0,
    refine_iters: int = 0,
):
    """Строит центроиды всех классов. Возвращает (classes, centroids), оба L2-нормированы."""
    train_embs = normalize_rows(train_embs)
    train_labels = np.asarray(train_labels)

    classes = np.unique(train_labels)
    centroids = []
    for cls in classes:
        cls_embs = train_embs[train_labels == cls]
        c = _centroid_one_class_soft(
            cls_embs,
            trim_ratio=trim_ratio,
            trim_power=trim_power,
            refine_iters=refine_iters,
            mode=trim_mode,
        )
        centroids.append(c)
    centroids = np.vstack(centroids)
    return classes, centroids


def predict_centroid(
    train_embs: np.ndarray,
    train_labels,
    query_embs: np.ndarray,
    trim_ratio: float = 0.0,
    trim_mode: str = "hard",
    trim_power: float = 4.0,
    refine_iters: int = 0,
):
    """Предсказание класса по cos-sim к центроидам.

    Возвращает (pred_labels, pred_scores, classes, centroids).
    """
    query_embs = normalize_rows(query_embs)
    classes, centroids = build_centroids(
        train_embs, train_labels,
        trim_ratio=trim_ratio, trim_mode=trim_mode,
        trim_power=trim_power, refine_iters=refine_iters,
    )

    sims = cosine_similarity(query_embs, centroids)
    best_idx = np.argmax(sims, axis=1)

    pred_labels = classes[best_idx]
    pred_scores = sims[np.arange(len(query_embs)), best_idx]
    return pred_labels, pred_scores, classes, centroids


# =============================================================================
# Nearest: top-k soft-voting (vectorized + sweep по k и T)
# =============================================================================

def _softvote_topk(sims: np.ndarray, train_labels: np.ndarray, k: int, temperature: float):
    """Голосование top-k соседей с softmax-температурой.

    Возвращает (pred_labels, pred_scores, votes_matrix, classes).
    """
    effective_k = min(int(k), sims.shape[1])
    # Топ-k индексов по убыванию cos-sim.
    topk_idx = np.argpartition(-sims, effective_k - 1, axis=1)[:, :effective_k]
    rows = np.arange(sims.shape[0])[:, None]
    topk_sims = sims[rows, topk_idx]

    # Softmax по top-k с температурой.
    w = topk_sims / float(temperature)
    w = w - w.max(axis=1, keepdims=True)
    w = np.exp(w)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)

    classes = np.unique(train_labels)
    label_to_col = {lab: j for j, lab in enumerate(classes)}

    Q = sims.shape[0]
    votes = np.zeros((Q, len(classes)), dtype=np.float32)

    topk_labels = train_labels[topk_idx]  # (Q, k)

    # Векторизованное накопление голосов: маппим labels → индексы классов
    # и аккумулируем веса через np.add.at по каждой колонке top-k.
    col_idx = np.vectorize(label_to_col.get)(topk_labels)  # (Q, k)
    for j in range(effective_k):
        np.add.at(votes, (np.arange(Q), col_idx[:, j]), w[:, j])

    best_col = np.argmax(votes, axis=1)
    pred_labels = classes[best_col]
    pred_scores = votes[np.arange(Q), best_col]
    return pred_labels, pred_scores, votes, classes


def predict_nearest(
    train_embs: np.ndarray,
    train_labels,
    query_embs: np.ndarray,
    k: int = 5,
    temperature: float = 0.1,
):
    """Одиночный прогон nearest с заданными (k, temperature)."""
    train_embs = normalize_rows(train_embs).astype(np.float32)
    query_embs = normalize_rows(query_embs).astype(np.float32)
    train_labels = np.asarray(train_labels)

    if k < 1 or len(train_embs) == 0:
        raise ValueError("k must be >= 1 and train set non-empty")

    sims = query_embs @ train_embs.T  # (Q, T)
    preds, scores, _, _ = _softvote_topk(sims, train_labels, k=k, temperature=temperature)
    return preds, scores


def predict_nearest_sweep(
    train_embs: np.ndarray,
    train_labels,
    query_embs: np.ndarray,
    k_list,
    t_list,
):
    """Sweep по (k, T) с переиспользованием матрицы sims.

    Матрица cos-sim (самая дорогая операция) считается один раз,
    после чего перебираются все комбинации k и temperature.
    Возвращает список dict с предсказаниями и весами для каждой пары (k, T).
    """
    train_embs = normalize_rows(train_embs).astype(np.float32)
    query_embs = normalize_rows(query_embs).astype(np.float32)
    train_labels = np.asarray(train_labels)

    sims = query_embs @ train_embs.T

    results = []
    for k_try in k_list:
        for t_try in t_list:
            preds, scores, votes, classes = _softvote_topk(
                sims, train_labels, k=int(k_try), temperature=float(t_try),
            )
            results.append({
                "k": int(k_try),
                "temperature": float(t_try),
                "pred_labels": preds,
                "pred_scores": scores,
                "votes": votes,
                "classes": classes,
            })
    return results


# =============================================================================
# Ensemble: centroid + nearest (взвешенная смесь scores)
# =============================================================================

def predict_centroid_nn_ensemble(
    train_embs: np.ndarray,
    train_labels,
    query_embs: np.ndarray,
    # centroid hyperparams
    trim_ratio: float = 0.15,
    trim_mode: str = "soft",
    trim_power: float = 4.0,
    refine_iters: int = 1,
    # nearest hyperparams
    k: int = 5,
    temperature: float = 0.1,
    # mix
    alpha: float = 0.5,
):
    """Ансамбль centroid + nearest со смешиванием в нормированном виде.

    score(query, class) = alpha * cos(query, centroid_class)
                        + (1 - alpha) * nn_softvote_weight(query, class)

    cos в [-1, 1], soft-vote weights в [0, 1]. Чтобы шкалы были сопоставимы,
    обе матрицы приводятся к [0, 1] через per-query min-max перед смешиванием.
    """
    train_embs_n = normalize_rows(train_embs).astype(np.float32)
    query_embs_n = normalize_rows(query_embs).astype(np.float32)
    train_labels = np.asarray(train_labels)

    # 1) Centroid scores.
    classes_c, centroids = build_centroids(
        train_embs_n, train_labels,
        trim_ratio=trim_ratio, trim_mode=trim_mode,
        trim_power=trim_power, refine_iters=refine_iters,
    )
    centroid_scores = query_embs_n @ centroids.T  # (Q, |C|)

    # 2) Nearest soft-vote scores.
    sims = query_embs_n @ train_embs_n.T
    _, _, nn_votes, classes_nn = _softvote_topk(
        sims, train_labels, k=k, temperature=temperature,
    )

    # Классы должны совпадать у обоих методов: они построены на одном train_labels.
    assert np.array_equal(classes_c, classes_nn), "Class sets must match between centroid and nn"

    # 3) Per-query min-max приведение к [0, 1] для совместимости шкал.
    def _rowwise_minmax(M):
        m = M.min(axis=1, keepdims=True)
        M2 = M - m
        rng = M2.max(axis=1, keepdims=True) + 1e-12
        return M2 / rng

    centroid_n = _rowwise_minmax(centroid_scores)
    nn_n = _rowwise_minmax(nn_votes)

    final = float(alpha) * centroid_n + (1.0 - float(alpha)) * nn_n
    best_col = np.argmax(final, axis=1)
    pred_labels = classes_c[best_col]
    pred_scores = final[np.arange(len(query_embs_n)), best_col]
    return pred_labels, pred_scores, classes_c
