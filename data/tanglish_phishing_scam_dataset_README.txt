Tanglish Phishing / Scam Message Dataset
=========================================

Files
-----
- tanglish_phishing_scam_dataset.csv: 30 labeled Tanglish examples.
- This README explains scope and provenance.

Schema
------
id, message, label, scam_category, red_flags, provenance

Label
-----
All records are labeled SCAM. This is a positive-only starter set, not a balanced
classification dataset. Add benign/legitimate messages before training a binary
classifier, and keep a separate test set.

Provenance and limitations
--------------------------
The dataset is primarily synthetic Tanglish text created for defensive awareness and
scam-detection prototyping. A few examples are explicitly described as Tanglish
adaptations of publicly documented scam patterns. They are NOT represented as exact
messages collected from real victims. Where public reporting supports a pattern, the
source URL is listed in the row's provenance field.

Public references consulted
---------------------------
1. Government of India / Press Information Bureau, DoT action on electricity KYC
   update scams (17 June 2024):
   https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=2025970&lang=2&reg=48
2. Tamil Nadu electricity bill scam reporting:
   https://www.news9live.com/state/tamil-nadu/electricity-bill-scam-on-the-rise-in-tamil-nadu-60-complaints-in-two-months-190449
3. Tamil Nadu courier / USSD call-forwarding scam warning:
   https://www.dtnext.in/news/tamilnadu/ussd-dial-the-emerging-cyber-fraud-warn-tn-cybercrime-cops
4. West Bengal Cyber Crime Wing, fake job offer SMS:
   https://cybercrimewing.wb.gov.in/FakeJobOfferSMS
5. Puducherry Police, Internet Safety User Handbook (parcel-delivery scam):
   https://police.py.gov.in/Internet%20Safety%20-%20User%20Safety%20Handbook.pdf
6. National Cyber Crime Reporting Portal advisories:
   https://cybercrime.gov.in/Webform/Advisory.aspx
7. https://github.com/mahalakshmidrv/ScamShield-AI/blob/main/data/processed/scam_dataset.csv

Safety
------
Links, phone numbers, and app files are represented with placeholders. Do not replace
placeholders with live malicious links or use this dataset to contact real people.
For actual incidents in India, report financial cyber fraud promptly via 1930 or
https://cybercrime.gov.in/.
