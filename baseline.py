import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from pathlib import Path

TRAIN_PATH = "data/train.csv"
# train.csv / train_paraphrase.csv / train_backtranslate.csv / train_augmented.csv
# train_augmented_summarized.csv / train_augmented_original_plus_summary.csv
TEST_PATH  = "data/test.csv"
TEXT_COL = "text"
LABEL_COL = "label"

df  = pd.read_csv(TRAIN_PATH)
print(df["label"].value_counts().sort_index())

def make_base_pipeline():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)$$[A-Z_]+$$|\b\w\w+\b",
        lowercase=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
    )
    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LinearSVC(class_weight="balanced")

    return Pipeline([
        ("tfidf", features),
        ("clf", clf),
    ])

def eval_model(model, X_test, y_test, ndigits=3):
    pred = model.predict(X_test)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
    }
    return {k: round(v, ndigits) for k, v in metrics.items()}

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df[TEXT_COL].astype(str)
    y_train = train_df[LABEL_COL].astype(str)
    X_test  = test_df[TEXT_COL].astype(str)
    y_test  = test_df[LABEL_COL].astype(str)

    model = make_base_pipeline()
    model.fit(X_train, y_train)

    metrics = eval_model(model, X_test, y_test)
    print("BASELINE METRICS:", metrics)

if __name__ == "__main__":
    main()

# baseline train.csv balanced_accuracy: 0.443, macro_f1: 0.44
# baseline train_paraphrase.csv balanced_accuracy: 0.462, macro_f1: 0.466
# baseline train_paraphrase_2.csv balanced_accuracy: 0.451, macro_f1: 0.454
# baseline train_paraphrase_3.csv balanced_accuracy: 0.463, macro_f1: 0.47
# baseline train_backtranslate.csv balanced_accuracy: 0.463, macro_f1: 0.472
# baseline train_backtranslate_2.csv balanced_accuracy: 0.451, macro_f1: 0.454
# baseline train_backtranslate_3.csv balanced_accuracy: 0.452, macro_f1: 0.454
# baseline train_augmented.csv balanced_accuracy: 0.463, macro_f1: 0.472
# baseline train_augmented_2.csv balanced_accuracy: 0.467, macro_f1: 0.477
# baseline train_augmented_3.csv balanced_accuracy: 0.463, macro_f1: 0.472

# baseline train_augmented_summarized.csv balanced_accuracy: 0.338, macro_f1: 0.361
# baseline train_augmented_original_plus_summary.csv balanced_accuracy: 0.463, macro_f1: 0.479
# baseline train_augmented_summarized_2.csv balanced_accuracy: 0.026, macro_f1: 0.007
# baseline train_augmented_original_plus_summary_2.csv balanced_accuracy: 0.457, macro_f1: 0.47
# baseline train_augmented_summarized_3.csv balanced_accuracy: 0.364, macro_f1: 0.376
# baseline train_augmented_original_plus_summary_3.csv balanced_accuracy: 0.457, macro_f1: 0.47

# MAX_LENGTH = 256
# BATCH_SIZE = 16
# NUM_EPOCHS = 10
# LR = 2e-5
# WEIGHT_DECAY = 0.01
# WARMUP_RATIO = 0.1
# rubert-base-case train.csv  balanced_accuracy: 0.304768, macro_f1: 0.295318
# rubert-base-case train_paraphrase.csv balanced_accuracy 0.369679, macro_f1: 0.377209
# rubert-base-cased train_paraphrase_2.csv balanced_accuracy: 0.340045, macro_f1: 0.336009
# rubert-base-case train_backtranslate.csv balanced_accuracy: 0.382728, macro_f1: 0.382456
# rubert-base-cased train_backtranslate_2.csv balanced_accuracy: 0.348204, macro_f1: 0.347102
# rubert-base-cased train_augmented.csv balanced_accuracy: 0.394003, macro_f1: 0.396188
# rubert-base-cased train_augmented_2.csv balanced_accuracy: 0.361901, macro_f1: 0.364837

# ---- классификация
        
# rubert-base-case AutoModel train.csv  balanced_accuracy: 0.345815, macro_f1: 0.341168
# rubert-base-case AutoModel train_paraphrase.csv balanced_accuracy 0.410021, macro_f1: 0.408506
# rubert-base-cased AutoModel train_paraphrase_2.csv balanced_accuracy: 0.418388, macro_f1: 0.429682
# rubert-base-cased AutoModel train_paraphrase_3.csv balanced_accuracy: 0.401685, macro_f1: 0.405676
# rubert-base-case AutoModel train_backtranslate.csv balanced_accuracy: 0.395779, macro_f1: 0.397109
# rubert-base-cased AutoModel train_backtranslate_2.csv balanced_accuracy: 0.372184, macro_f1: 0.373936
# rubert-base-cased AutoModel train_backtranslate_3.csv balanced_accuracy: 0.419235, macro_f1: 0.419539
# rubert-base-cased AutoModel train_augmented.csv balanced_accuracy: 0.426671, macro_f1: 0.426301
# rubert-base-cased AutoModel train_augmented_2.csv balanced_accuracy: 0.395242, macro_f1: 0.396828
# rubert-base-cased AutoModel train_augmented_3.csv balanced_accuracy: 0.433344, macro_f1: 0.426494
        
# rubert-base-case meanPooling train.csv  balanced_accuracy: 0.353825, macro_f1: 0.357340
# rubert-base-case meanPooling train_paraphrase.csv balanced_accuracy 0.395913, macro_f1: 0.391975
# rubert-base-cased meanPooling train_paraphrase_2.csv balanced_accuracy: 0.429026, macro_f1: 0.420598
# rubert-base-cased meanPooling train_paraphrase_3.csv balanced_accuracy: 0.443284, macro_f1: 0.437885
# rubert-base-case meanPooling train_backtranslate.csv balanced_accuracy: 0.401737, macro_f1: 0.400306
# rubert-base-cased meanPooling train_backtranslate_2.csv balanced_accuracy: 0.447853, macro_f1: 0.457282
# rubert-base-cased meanPooling train_backtranslate_3.csv balanced_accuracy: 0.455458, macro_f1: 0.460190
# rubert-base-cased meanPooling train_augmented.csv balanced_accuracy: 0.435280, macro_f1: 0.431922
# rubert-base-cased meanPooling train_augmented_2.csv balanced_accuracy: 0.450676, macro_f1: 0.452683
# rubert-base-cased meanPooling train_augmented_3.csv balanced_accuracy: 0.448233, macro_f1: 0.453793
       
# ruRoberta-large AutoModel train.csv  balanced_accuracy: 0.394214, macro_f1: 0.399566
# ruRoberta-large AutoModel train_paraphrase.csv balanced_accuracy 0.468766, macro_f1: 0.470790
# ruRoberta-large AutoModel train_paraphrase_2.csv balanced_accuracy: 0.027778, macro_f1: 0.006495
# ruRoberta-large AutoModel train_paraphrase_3.csv balanced_accuracy: 0.474551, macro_f1: 0.486725
# ruRoberta-large AutoModel train_backtranslate.csv balanced_accuracy: 0.119339, macro_f1: 0.111930
# ruRoberta-large AutoModel train_backtranslate_2.csv balanced_accuracy: 0.416208, macro_f1: 0.426317
# ruRoberta-large AutoModel train_backtranslate_3.csv balanced_accuracy: 0.467185, macro_f1: 0.465174
# ruRoberta-large AutoModel train_augmented.csv balanced_accuracy: 0.456992, macro_f1: 0.471013
# ruRoberta-large AutoModel train_augmented_2.csv balanced_accuracy: 0.247079, macro_f1: 0.242977
# ruRoberta-large AutoModel train_augmented_3.csv balanced_accuracy: 0.474348, macro_f1: 0.466214
             
# ruRoberta-large meanPooling train.csv  balanced_accuracy: 0.426209, macro_f1: 0.444748
# ruRoberta-large meanPooling train_paraphrase.csv balanced_accuracy 0.460898, macro_f1: 0.466976
# ruRoberta-large meanPooling train_paraphrase_2.csv balanced_accuracy: 0.488878, macro_f1: 0.487119
# ruRoberta-large meanPooling train_paraphrase_3.csv balanced_accuracy: 0.466266, macro_f1: 0.463188
# ruRoberta-large meanPooling train_backtranslate.csv balanced_accuracy: 0.432684, macro_f1: 0.423993
# ruRoberta-large meanPooling train_backtranslate_2.csv balanced_accuracy: 0.453921, macro_f1: 0.454833
# ruRoberta-large meanPooling train_backtranslate_3.csv balanced_accuracy: 0.478354, macro_f1: 0.487375
# ruRoberta-large meanPooling train_augmented.csv balanced_accuracy: 0.476920, macro_f1: 0.487293
# ruRoberta-large meanPooling train_augmented_2.csv balanced_accuracy: 0.481015, macro_f1: 0.474783
# ruRoberta-large meanPooling train_augmented_3.csv balanced_accuracy: 0.464578, macro_f1: 0.483161
                 
# bert-classification train_backtranslate_2.csv balanced_accuracy: 0.441557, macro_f1: 0.450440
# bert-classification train_backtranslate_3.csv balanced_accuracy: 0.388507, macro_f1: 0.382487
# bert-classification train_augmented_3 balanced_accuracy: 0.463967, macro_f1: 0.470460 
# bert-classification train_augmented_original_plus_summary_3 balanced_accuracy: 0.027778, macro_f1: 0.006495

# --- Композитный вектор

# python -m hybrid.main build \
#   --train /Users/v.papadyk/ml/mifi-vkr/data/train_augmented_3.csv \
#   --test /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
#   --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
#   --model-dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
#   --device cpu

# python -m hybrid.main build \
#   --train /Users/v.papadyk/ml/mifi-vkr/data/train_augmented_3.csv \
#   --test /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
#   --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
#   --device cpu

# python -m hybrid.main classical \
#   --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec

# train_augmented
# linear_svc {'balanced_accuracy': 0.340786, 'macro_f1': 0.321917}
# logreg {'balanced_accuracy': 0.390829, 'macro_f1': 0.389622}
# fine-tune train_augmented_3
# linear_svc: {'balanced_accuracy': 0.155629, 'macro_f1': 0.152774}
# logreg: {'balanced_accuracy': 0.270957, 'macro_f1': 0.27713}
# original train_augmented_3 weight 5
# linear_svc: {'balanced_accuracy': 0.288799, 'macro_f1': 0.302895}
# logreg: {'balanced_accuracy': 0.300821, 'macro_f1': 0.300796
# original train_augmented_3 weight 50
# linear_svc: {'balanced_accuracy': 0.221075, 'macro_f1': 0.217713}
# logreg: {'balanced_accuracy': 0.343632, 'macro_f1': 0.356956}

# python -m hybrid.main mlp \
#   --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
#   --epochs 40 \
#   --device cpu

# train_augmented
# balanced_accuracy: 0.447552, macro_f1: 0.433175
# fine-tune train_augmented_3:
# balanced_accuracy: 0.32539, macro_f1: 0.345667
# original train_augmented_3
# weight 5 balanced_accuracy: 0.387764, macro_f1: 0.400307
# weight 50 balanced_accuracy: 0.372608, macro_f1: 0.397246
# ---- Косинусное сходство

# python -m cosine_similarity_classification.main \
#   --train_file /Users/v.papadyk/ml/mifi-vkr/data/train_augmented_3.csv \
#   --test_file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
#   --model_dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
#   --method centroid

# python -m cosine_similarity_classification.main \
#   --train_file /Users/v.papadyk/ml/mifi-vkr/data/train_augmented_3.csv \
#   --test_file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
#   --model_dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
#   --method nearest

# centroid train_augmented: balanced_accuracy: 0.149607, macro_f1: 0.121122
# centroid train_augmented_3_v1: balanced_accuracy: 0.118086, macro_f1: 0.082073
# centroid train_augmented_3_v2: balanced_accuracy: ?, macro_f1: ?
# nearest train_augmented: balanced_accuracy: 0.246947, macro_f1: 0.250962
# nearest train_augmented_3_v1: balanced_accuracy: 0.238407, macro_f1: 0.217342
# nearest train_augmented_3_v2: balanced_accuracy: ?, macro_f1: ?

# ---- Суммаризация

# IlyaGusev/rut5_base_sum_gazeta:
# rubert-base-cased AutoModel train_augmented_summarized.csv balanced_accuracy: 0.223529, macro_f1: 0.228690
# rubert-base-cased AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.434526, macro_f1: 0.443603
# rubert-base-case meanPooling train_augmented_summarized.csv balanced_accuracy: 0.220817, macro_f1: 0.221949
# rubert-base-case meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.429666, macro_f1: 0.430774
# ruRoberta-large AutoModel train_augmented_summarized.csv balanced_accuracy: 0.333970, macro_f1: 0.332734
# ruRoberta-large AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.475647, macro_f1: 0.468746
# ruRoberta-large meanPooling train_augmented_summarized.csv balanced_accuracy: 0.379292, macro_f1: 0.384017
# ruRoberta-large meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.446830, macro_f1: 0.466032
       
# IlyaGusev/mbart_ru_sum_gazeta:
# rubert-base-cased AutoModel train_augmented_summarized.csv balanced_accuracy: 0.027778, macro_f1: 0.001598
# rubert-base-cased AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.398335, macro_f1: 0.399363
# rubert-base-case meanPooling train_augmented_summarized.csv balanced_accuracy: 0.034722, macro_f1: 0.001750
# rubert-base-case meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.404449, macro_f1: 0.402429
# ruRoberta-large AutoModel train_augmented_summarized.csv balanced_accuracy: 0.023641, macro_f1: 0.005958
# ruRoberta-large AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.439066, macro_f1: 0.435186
# ruRoberta-large meanPooling train_augmented_summarized.csv balanced_accuracy: 0.027778, macro_f1: 0.006495
# ruRoberta-large meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.027778, macro_f1: 0.006495
                                                 
# d0rj/rut5-base-summ:
# rubert-base-cased AutoModel train_augmented_summarized.csv balanced_accuracy: 0.131789, macro_f1: 0.116513
# rubert-base-cased AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.432433, macro_f1: 0.440640              
# rubert-base-case meanPooling train_augmented_summarized.csv balanced_accuracy: 0.211926, macro_f1: 0.194112
# rubert-base-case meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.450357, macro_f1: 0.441125
# ruRoberta-large AutoModel train_augmented_summarized.csv balanced_accuracy: 0.317597, macro_f1: 0.289931
# ruRoberta-large AutoModel train_augmented_original_plus_summary.csv balanced_accuracy: 0.454132, macro_f1: 0.465529
# ruRoberta-large meanPooling train_augmented_summarized.csv balanced_accuracy: 0.347784, macro_f1: 0.335338
# ruRoberta-large meanPooling train_augmented_original_plus_summary.csv balanced_accuracy: 0.027778, macro_f1: 0.006495
            