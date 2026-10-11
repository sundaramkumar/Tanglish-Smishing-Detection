"""Experiments 1-3 (Sections 4 and 5 of the paper).

Requires results/audit_rows.csv, so run src/audit.py first.

  Experiment 1  Code-mixing alone. 5-fold stratified CV on UCI. In each test
                fold, recall on English UCI spam that has a Tanglish
                translation, and on those translations (Table 3).
  Experiment 2  Train on all of UCI, test on groups A, B and D, with 95% Wilson
                intervals (Table 4) and a breakdown by scam type (Table 5).
  Experiment 3  Add Tanglish scams to training, hold out one group, measure
                held-out recall and the benign probes flagged (Table 6); top
                spam-weighted features and probe decision scores (Section 5.3).

Outputs (in results/)
  results.json   every number reported in Section 5
  tables.md      Tables 3-6 as they appear in the paper

Run:  python src/experiments.py
"""
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from common import AUDIT_CSV, DECISIONS_CSV, RESULTS, SEED, load_uci, models

warnings.filterwarnings("ignore")
INDIAN_GROUPS = ["A_seed", "B_callscript", "D_template"]


def wilson(x, n, z=1.96):
    """95% Wilson score interval for x successes out of n, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h) * 100, 1), round(min(1.0, c + h) * 100, 1))


def scam_type(category):
    """Coarse scam type from the free-text scam_category field (Table 5)."""
    s = str(category).lower()
    for key, label in [("electric", "Electricity"), ("courier", "Courier / parcel"),
                       ("parcel", "Courier / parcel"), ("postal", "Courier / parcel"),
                       ("job", "Job / task"), ("task", "Job / task"), ("upi", "UPI / payment"),
                       ("kyc", "KYC / bank account"), ("bank", "KYC / bank account"),
                       ("card", "KYC / bank account"), ("otp", "KYC / bank account"),
                       ("prize", "Prize / lottery"), ("lottery", "Prize / lottery"),
                       ("govt", "Government / police"), ("government", "Government / police"),
                       ("authority", "Government / police"), ("police", "Government / police")]:
        if key in s:
            return label
    return "Other"


def pct(v):
    return f"{100 * v:.1f}%"


def main():
    R = {}
    u = load_uci()
    ud = load_uci(dedup=True)
    R["uci"] = dict(rows=len(u), after_dedup=len(ud), spam=int(ud.y.sum()), ham=int((1 - ud.y).sum()))

    t = pd.read_csv(AUDIT_CSV)
    decisions = pd.read_csv(DECISIONS_CSV)
    removed = decisions.loc[decisions.decision == "remove", "id"].tolist()
    t = t.drop_duplicates("message")
    probes = t[t.id.isin(removed)].sort_values("id")
    t = t[~t.id.isin(removed)]
    R["corpus_after_audit"] = len(t)
    R["benign_probes"] = probes.id.tolist()

    # English-Tanglish pairs: group C rows linked to a UCI spam message, one per source
    pos = dict(zip(ud.msg, ud.index))
    c = t[(t.group == "C_uci") & (t.uci_idx >= 0) & (t.uci_label == "spam")]
    c = c.assign(ud_idx=[pos[u.msg[int(i)]] for i in c.uci_idx]).drop_duplicates("ud_idx")
    R["pairs"] = len(c)

    # ------------------------------------------------------------- Experiment 1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = {k: dict(en_f1=[], en_fpr=[], en_rec=[], ta_rec=[], n_pairs=[]) for k in models()}
    for tr, te in skf.split(ud.msg, ud.y):
        cp = c[c.ud_idx.isin(set(te))]
        for k, m in models().items():
            m.fit(ud.msg[tr], ud.y[tr])
            pe, ye = m.predict(ud.msg[te]), ud.y[te].values
            acc[k]["en_f1"].append(f1_score(ye, pe))
            acc[k]["en_fpr"].append(((pe == 1) & (ye == 0)).sum() / (ye == 0).sum())
            acc[k]["en_rec"].append(m.predict(ud.msg[cp.ud_idx]).mean())
            acc[k]["ta_rec"].append(m.predict(cp.message).mean())
            acc[k]["n_pairs"].append(len(cp))
    R["exp1"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in acc.items()}
    print("Experiment 1 done")

    # ------------------------------------------------------------- Experiment 2
    ind = t[t.group.isin(INDIAN_GROUPS)]
    fitted, R["exp2"] = {}, {}
    for k, m in models().items():
        m.fit(ud.msg, ud.y)
        fitted[k] = m
        p = m.predict(ind.message)
        row = {}
        for g in INDIAN_GROUPS + ["all"]:
            mask = np.ones(len(ind), bool) if g == "all" else (ind.group == g).values
            x, n = int(p[mask].sum()), int(mask.sum())
            row[g] = dict(caught=x, n=n, recall=x / n, ci95=wilson(x, n))
        row["benign_probes_flagged"] = int(m.predict(probes.message).sum())
        R["exp2"][k] = row

    ab = ind[ind.group != "D_template"].copy()
    ab["type"] = ab.scam_category.map(scam_type)
    for k in ["SVM (word)", "SVM (char 2-5)"]:
        ab[k] = fitted[k].predict(ab.message)
    R["exp2_by_type"] = (ab.groupby("type")
                         .agg(n=("message", "size"), svm_word=("SVM (word)", "mean"),
                              svm_char=("SVM (char 2-5)", "mean"))
                         .sort_values("n", ascending=False).reset_index().to_dict("records"))

    svm = fitted["SVM (word)"]
    voc, coef = svm.steps[0][1].vocabulary_, svm.steps[1][1].coef_[0]
    R["uci_svm_weights"] = {w: (round(float(coef[voc[w]]), 3) if w in voc else None)
                            for w in ["kyc", "aadhaar", "upi", "otp", "parcel", "fee", "account",
                                      "bank", "card", "prize", "won", "job", "electricity"]}
    names = svm.steps[0][1].get_feature_names_out()
    R["top_features_uci_only"] = names[np.argsort(-coef)[:25]].tolist()
    print("Experiment 2 done")

    # ------------------------------------------------------------- Experiment 3
    pools = {"A_seed": t[t.group == "A_seed"], "B_callscript": t[t.group == "B_callscript"],
             "D_template": t[t.group == "D_template"], "C_uci": c}
    R["exp3"] = {}
    for held in INDIAN_GROUPS:
        add = pd.concat([pools[g] for g in pools if g != held])
        X = pd.concat([ud.msg, add.message])
        y = np.r_[ud.y.values, np.ones(len(add), int)]
        R["exp3"][held] = {}
        for k in ["MNB (word)", "SVM (word)", "SVM (char 2-5)"]:
            m = models()[k].fit(X, y)
            R["exp3"][held][k] = dict(recall=float(m.predict(pools[held].message).mean()),
                                      benign_probes_flagged=int(m.predict(probes.message).sum()))

    add = pd.concat(pools.values())
    adapted = models()["SVM (word)"].fit(pd.concat([ud.msg, add.message]),
                                         np.r_[ud.y.values, np.ones(len(add), int)])
    an = adapted.steps[0][1].get_feature_names_out()
    R["top_features_adapted"] = an[np.argsort(-adapted.steps[1][1].coef_[0])[:25]].tolist()
    R["probe_decision_scores"] = {
        "ids": probes.id.tolist(),
        "uci_only": svm.decision_function(probes.message).round(2).tolist(),
        "with_tanglish_scams": adapted.decision_function(probes.message).round(2).tolist(),
    }
    print("Experiment 3 done")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(R, indent=1, default=str))
    write_tables(R)
    print((RESULTS / "tables.md").read_text())


def write_tables(R):
    L = ["# Results tables (generated by src/experiments.py)\n"]
    L += ["## Table 3: Recall on paired English and Tanglish versions of the same scams (5-fold CV)\n",
          "| Model | UCI F1 | UCI ham FPR | Recall, English | Recall, Tanglish | Change (points) |",
          "|---|---|---|---|---|---|"]
    for k, v in R["exp1"].items():
        L.append(f"| {k} | {v['en_f1']:.3f} | {100 * v['en_fpr']:.2f}% | {pct(v['en_rec'])} | "
                 f"{pct(v['ta_rec'])} | {100 * (v['ta_rec'] - v['en_rec']):+.1f} |")
    L += ["", f"Pairs: {R['pairs']} (mean per test fold: "
              f"{np.mean([v['n_pairs'] for v in R['exp1'].values()]):.1f})\n"]

    L += ["## Table 4: Recall on Indian-pattern Tanglish scams, trained on UCI only (95% Wilson interval)\n",
          "| Model | A. Seed | B. Call scripts | D. Templates | All |", "|---|---|---|---|---|"]
    for k, v in R["exp2"].items():
        cells = [f"{pct(v[g]['recall'])} [{v[g]['ci95'][0]}-{v[g]['ci95'][1]}] ({v[g]['caught']}/{v[g]['n']})"
                 for g in INDIAN_GROUPS + ["all"]]
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    L += ["", "## Table 5: Recall by scam type, groups A and B, trained on UCI only\n",
          "| Scam type | Messages | SVM (word) | SVM (char 2-5) |", "|---|---|---|---|"]
    for r in R["exp2_by_type"]:
        L.append(f"| {r['type']} | {r['n']} | {pct(r['svm_word'])} | {pct(r['svm_char'])} |")

    L += ["", "## Table 6: Held-out group recall and benign probes flagged after adding Tanglish scams\n",
          "| Held-out group | MNB (word) | SVM (word) | SVM (char 2-5) | Probes flagged (of 4) |",
          "|---|---|---|---|---|"]
    ks = ["MNB (word)", "SVM (word)", "SVM (char 2-5)"]
    for g, v in R["exp3"].items():
        L.append(f"| {g} | " + " | ".join(pct(v[k]["recall"]) for k in ks) + " | "
                 + " / ".join(str(v[k]["benign_probes_flagged"]) for k in ks) + " |")
    L.append("| UCI only (reference) | - | - | - | "
             + " / ".join(str(R["exp2"][k]["benign_probes_flagged"]) for k in ks) + " |")

    s = R["probe_decision_scores"]
    L += ["", "## Section 5.3 details\n",
          "Top 25 spam-weighted features, word SVM, UCI + all Tanglish scams:  "
          + ", ".join(R["top_features_adapted"]), "",
          "Top 25 spam-weighted features, word SVM, UCI only:  " + ", ".join(R["top_features_uci_only"]), "",
          "Benign probe decision scores (" + ", ".join(s["ids"]) + "):  UCI only "
          + str(s["uci_only"]) + "  ->  with Tanglish scams " + str(s["with_tanglish_scams"]), "",
          "UCI-only word SVM weights (Section 5.2):  "
          + ", ".join(f"{w} {c}" for w, c in R["uci_svm_weights"].items())]
    (RESULTS / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
