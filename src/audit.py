"""Audit of the Tanglish scam corpus (Section 3 of the paper).

What it does
  1. Assigns every row to a provenance group (A seed, B call scripts,
     C UCI translations, D templates).
  2. Links each row to its source message in the UCI SMS Spam Collection:
       pass 1  a shared digit string of 5+ characters or a shared URL
       pass 2  TF-IDF cosine similarity >= 0.35 over English tokens
  3. Flags exact duplicate messages.
  4. Lists rows whose closest UCI source is a *benign* (ham) message, so they
     can be checked by hand. The decisions taken are recorded in
     data/audit_decisions.csv and are read by experiments.py.
  5. Reports the share of word tokens never seen in UCI (Table 1).

Outputs (in results/)
  audit_rows.csv               one line per corpus row with group and UCI match
  audit_benign_candidates.csv  rows to review by hand
  audit_summary.json           counts used in Table 1 and Section 3.2

Run:  python src/audit.py
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (AUDIT_CSV, DECISIONS_CSV, GROUP_NAMES, RESULTS, group_of,
                    load_tanglish, load_uci)

SIM_THRESHOLD = 0.35          # pass-2 similarity threshold
BENIGN_REVIEW_THRESHOLD = 0.30  # ham similarity above which a row is listed for review


def link_keys(text):
    """Digit strings of 5+ characters (spaces removed) and URLs."""
    s = str(text)
    digits = set(re.findall(r"\d{5,}", s.replace(" ", "")))
    urls = set(re.findall(r"(?:www\.|http)\S+", s.lower()))
    return digits | urls


def main():
    RESULTS.mkdir(exist_ok=True)
    t = load_tanglish()
    u = load_uci()
    summary = {"rows": len(t), "labels": t["label"].value_counts().to_dict()}

    # ---- 1. provenance groups
    t["group"] = t["id"].map(group_of)

    # ---- 2. link to UCI
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]{3,}|\d{4,}")
    vec.fit(pd.concat([u.msg, t.message]))
    S = cosine_similarity(vec.transform(t.message), vec.transform(u.msg))
    uci_keys = [link_keys(m) for m in u.msg]

    match, idx, sim = [], [], []
    for i, msg in enumerate(t.message):
        keys = link_keys(msg)
        j = int(S[i].argmax())
        how = ""
        if keys:
            cand = [jj for jj, kk in enumerate(uci_keys) if keys & kk]
            if cand:
                j = max(cand, key=lambda jj: S[i, jj])
                how = "key"
        if not how and S[i, j] >= SIM_THRESHOLD:
            how = "sim"
        match.append(how)
        idx.append(j if how else -1)
        sim.append(float(S[i, j]))
    t["match"] = match
    t["uci_idx"] = idx
    t["sim"] = np.round(sim, 4)
    t["uci_label"] = [u.label[j] if j >= 0 else "" for j in t.uci_idx]
    t["uci_msg"] = [u.msg[j] if j >= 0 else "" for j in t.uci_idx]

    # ---- 3. duplicates
    t["duplicate_of_earlier_row"] = t.message.duplicated()

    # ---- 4. benign-origin candidates in group C
    ham = u[u.label == "ham"].reset_index(drop=True)
    spam = u[u.label == "spam"].reset_index(drop=True)
    c = t[t.group == "C_uci"]
    sh = cosine_similarity(vec.transform(c.message), vec.transform(ham.msg))
    ss = cosine_similarity(vec.transform(c.message), vec.transform(spam.msg))
    cand = c.assign(best_ham_sim=sh.max(1).round(3), best_spam_sim=ss.max(1).round(3),
                    closest_ham=[ham.msg[j] for j in sh.argmax(1)])
    cand = cand[(cand.match != "key") & (cand.best_ham_sim > cand.best_spam_sim)]
    cand = cand.drop_duplicates("message")
    cand["strong_candidate"] = cand.best_ham_sim >= BENIGN_REVIEW_THRESHOLD
    cand[["id", "message", "best_ham_sim", "best_spam_sim", "closest_ham", "strong_candidate"]] \
        .to_csv(RESULTS / "audit_benign_candidates.csv", index=False)

    # ---- 5. vocabulary unseen in UCI
    ud = load_uci(dedup=True)
    vocab = set(TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b").fit(ud.msg).vocabulary_)

    def oov(s):
        toks = re.findall(r"[a-zA-Z]{2,}", s.lower())
        return np.mean([w not in vocab for w in toks]) if toks else 0.0

    decisions = pd.read_csv(DECISIONS_CSV)
    removed = set(decisions.loc[decisions.decision == "remove", "id"])
    clean = t.drop_duplicates("message")
    clean = clean[~clean.id.isin(removed)]

    table1 = []
    for g, name in GROUP_NAMES.items():
        sub = t[t.group == g]
        table1.append({
            "group": name,
            "rows": int(len(sub)),
            "after_audit": int((clean.group == g).sum()),
            "mean_words": round(float(sub.message.str.split().str.len().mean()), 1),
            "unseen_in_uci_pct": round(100 * float(np.mean([oov(s) for s in sub.message])), 1),
        })

    c_clean = clean[(clean.group == "C_uci") & (clean.uci_idx >= 0) & (clean.uci_label == "spam")]
    pos = dict(zip(ud.msg, ud.index))
    pairs = len({pos[u.msg[int(j)]] for j in c_clean.uci_idx})

    summary.update({
        "groups": t.group.value_counts().to_dict(),
        "group_C_match_types": t[t.group == "C_uci"].match.replace("", "none").value_counts().to_dict(),
        "exact_duplicates": int(t.duplicate_of_earlier_row.sum()),
        "benign_candidates_listed": int(len(cand)),
        "strong_benign_candidates": cand.loc[cand.strong_candidate, "id"].tolist(),
        "removed_by_decision": sorted(removed),
        "rows_after_audit": int(len(clean)),
        "english_tanglish_pairs": int(pairs),
        "uci_spam_unseen_pct": round(100 * float(np.mean([oov(s) for s in ud.msg[ud.y == 1]])), 1),
        "table1": table1,
    })
    t.to_csv(AUDIT_CSV, index=False)
    (RESULTS / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    print(f"Rows: {summary['rows']}   exact duplicates: {summary['exact_duplicates']}")
    print(f"Group C linked by key: {summary['group_C_match_types'].get('key', 0)}")
    print(f"Strong benign-origin candidates: {summary['strong_benign_candidates']}")
    print(f"Removed (data/audit_decisions.csv): {summary['removed_by_decision']}")
    print(f"Rows after audit: {summary['rows_after_audit']}   English-Tanglish pairs: {pairs}")
    print(f"Words unseen in UCI, English UCI spam: {summary['uci_spam_unseen_pct']}%")
    print("\nTable 1")
    print(pd.DataFrame(table1).to_string(index=False))


if __name__ == "__main__":
    main()
