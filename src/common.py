"""Shared paths, loaders and model definitions for the Tanglish smishing study."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TANGLISH_CSV = DATA / "tanglish_phishing_scam_dataset.csv"
UCI_FILE = DATA / "SMSSpamCollection"
AUDIT_CSV = RESULTS / "audit_rows.csv"
DECISIONS_CSV = DATA / "audit_decisions.csv"

SEED = 42
GROUP_NAMES = {
    "A_seed": "A. Seed",
    "B_callscript": "B. Call scripts",
    "C_uci": "C. UCI translations",
    "D_template": "D. Templates",
}


def load_tanglish():
    t = pd.read_csv(TANGLISH_CSV, encoding="utf-8-sig")
    t["label"] = t["label"].str.upper()
    return t


def load_uci(dedup=False):
    """UCI SMS Spam Collection (Almeida et al., 2011). Tab-separated: label, text."""
    u = pd.read_csv(UCI_FILE, sep="\t", header=None, names=["label", "msg"], quoting=3)
    u["y"] = (u["label"] == "spam").astype(int)
    if dedup:
        u = u.drop_duplicates("msg").reset_index(drop=True)
    return u


def group_of(row_id):
    """Provenance group from the id range.

    The ranges follow the order in which the four sources were appended to the
    CSV and agree with the provenance field where it is filled in:
      TX001-TX035                       A: seed messages written to documented patterns
      TX036-089, TX274-297, TX655-665   B: Tanglish renderings of ScamShield-AI scripts
      TX666-TX754                       D: template-generated variants (lower-case label)
      everything else                   C: Tanglish translations of UCI spam
    """
    n = int(str(row_id)[2:])
    if n <= 35:
        return "A_seed"
    if n >= 666:
        return "D_template"
    if 36 <= n <= 89 or 274 <= n <= 297 or 655 <= n <= 665:
        return "B_callscript"
    return "C_uci"


def models():
    """The five classifiers reported in the paper (fresh, unfitted instances)."""
    w = dict(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    return {
        "MNB (word)": make_pipeline(TfidfVectorizer(**w), MultinomialNB(alpha=0.1)),
        "LR (word)": make_pipeline(TfidfVectorizer(**w),
                                   LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
        "SVM (word)": make_pipeline(TfidfVectorizer(**w), LinearSVC(C=1, class_weight="balanced")),
        "RF (word)": make_pipeline(TfidfVectorizer(**w),
                                   RandomForestClassifier(n_estimators=300, random_state=0,
                                                          n_jobs=-1, class_weight="balanced")),
        "SVM (char 2-5)": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
            LinearSVC(C=1, class_weight="balanced")),
    }
