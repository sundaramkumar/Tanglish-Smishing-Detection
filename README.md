# Tanglish Smishing Detection: Data and Code

Code and data for the paper "Tanglish Smishing Detection: A Tamil-English Code-Mixed Scam SMS Dataset and an Evaluation of Standard Spam Classifiers" (Kumar Sundaram).

Running `run_all.sh` reproduces every number, table and figure in the paper in about a minute on a laptop.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh                    # or run the three scripts below in order
```

| Step | Command | Produces | Paper section |
|---|---|---|---|
| 1 | `python3 src/audit.py` | `results/audit_rows.csv`, `audit_benign_candidates.csv`, `audit_summary.json` | Section 3, Table 1 |
| 2 | `python3 src/experiments.py` | `results/results.json`, `results/tables.md` | Sections 4 and 5, Tables 3 to 6 |
| 3 | `python3 src/make_figure.py` | `results/figure1_recall_by_group.png` | Figure 1 |

Step 2 reads the output of step 1, so run them in order.

## Folder layout

```
data/
  tanglish_phishing_scam_dataset.csv         the 754-message corpus
  tanglish_phishing_scam_dataset_README.txt  scope, schema and provenance notes
  audit_decisions.csv                        hand decisions taken after the audit
  SMSSpamCollection                          UCI SMS Spam Collection (training data)
src/
  common.py        paths, loaders, provenance groups, the five classifiers
  audit.py         provenance groups, UCI linking, duplicates, benign-origin check
  experiments.py   Experiments 1 to 3
  make_figure.py   Figure 1
results/           created by the scripts
```

## The corpus

`tanglish_phishing_scam_dataset.csv` has six columns: `id, message, label, scam_category, red_flags, provenance`. All 754 rows are labelled SCAM. Links, phone numbers and app files in the seed messages are replaced by placeholders such as `[suspicious link]`.

The messages are based on scam patterns documented in Department of Telecommunications (DoT) alerts, police advisories and news reports of recent frauds in Tamil Nadu. None of them is an intercepted SMS received by an identified person: they were written, adapted, translated or generated to follow those patterns. The corpus therefore tests whether a classifier recognises Tanglish versions of known scam patterns. It does not measure how often those patterns occur in real Tamil Nadu inboxes, and the wording of real messages may differ.

The rows come from four sources, identified by id range (`group_of` in `src/common.py`):

| Group | Ids | Source | Rows |
|---|---|---|---|
| A. Seed | TX001-TX035 | Written in Tanglish to match scam patterns in DoT, police and news advisories (URL in the `provenance` field) | 35 |
| B. Call scripts | TX036-089, TX274-297, TX655-665 | Tanglish renderings of scripts from the ScamShield-AI repository, whose data is itself synthetic | 89 |
| C. UCI translations | all other ids up to TX654 | Tanglish translations of UCI SMS Spam Collection messages | 541 |
| D. Templates | TX666-TX754 | Generated from templates; links use reserved example domains | 89 |

The audit finds 54 exact duplicates and four translations of benign UCI messages that carry a scam label (TX100, TX118, TX119, TX136). These four are removed from the scam set and used as benign probes, which leaves 696 messages. `data/audit_decisions.csv` records each decision and the reason for it. To change a decision, edit that file and rerun both scripts.

The corpus has no benign Tanglish messages apart from the four probes. Section 5.3 of the paper explains why that matters.

## The UCI SMS Spam Collection

`data/SMSSpamCollection` is the unmodified file from the UCI Machine Learning Repository (5,574 messages, 747 spam), included here so the results can be reproduced without a download. It is released under CC BY 4.0. Cite:

T.A. Almeida, J.M. Gómez Hidalgo, A. Yamakami, Contributions to the Study of SMS Spam Filtering: New Collection and Results, Proceedings of the 11th ACM Symposium on Document Engineering, 2011, 259-262. https://doi.org/10.1145/2034691.2034742

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Reproducibility notes

- Random seeds are fixed: `StratifiedKFold(random_state=42)` and `RandomForestClassifier(random_state=0)`. The linear models and naive Bayes are deterministic.
- The published results were produced with Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1 and matplotlib 3.11.2. Other versions should give the same numbers for MNB, LR and the SVMs. Random forest figures may shift by a fraction of a point with a different scikit-learn version.
- `results/tables.md` prints each table in the same layout as the paper, so the two can be compared line by line.

## Safety

Do not replace placeholders with live malicious links, and do not use the corpus to contact real people. In India, report financial cyber fraud by calling 1930 or at https://cybercrime.gov.in/.

## Licence

Data: CC BY 4.0
Code: MIT
