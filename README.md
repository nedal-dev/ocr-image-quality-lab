# OCR Image Quality Lab

An original synthetic experiment by Nedal / Idraaak. Three original English
passages, three image conditions each, nine actual Tesseract runs. The design in
plan.json is frozen before the main run. No settings are chosen from the results.

## Exact free environment

Ubuntu 24.04, Python 3.12, Pillow12.3.0. Install:

```sh
sudo apt-get update
sudo apt-get install tesseract-ocr=5.3.4-1build5 tesseract-ocr-eng=1:4.1.0-2 fonts-dejavu-core=2.37-8
python -m pip install -r requirements.txt
python experiment.py
python build_report.py
python -m http.server 8765
```

All cases use Tesseract5.3.4, bundled eng.traineddata, OEM1, PSM6, --dpi300,
OMP_THREAD_LIMIT=1. The same nominal DPI metadata is saved in every PNG.
Exact engine/Leptonica versions, OS, Python, font/model hashes and package
versions are recorded before recognition in data/environment.json.

DejaVu Sans2.37 regular,32px, black text on white;1600x1000 canvas with80px
margins and48px line spacing. Low resolution is400x250 via LANCZOS; tilt is
counterclockwise3degrees via BICUBIC, expand=true, white fill. No cropping,
sharpening, contrast alteration, hand correction or external OCR API.

## Evidence and CER

- texts.json and references/: original fictional passages;93,95,94 words.
- images/: all9actual inputs, lossless PNG.
- raw/: exact Tesseract text outputs and stderr, uncorrected.
- normalized/: reference and OCR strings actually scored.
- data/results.csv and results.json: all9rows, dimensions, counts and CER.
- data/edit-operations.json: complete minimum-distance edit alignments.
- data/manifest.json: evidence file sizes and SHA-256 hashes.
- fonts/ and models/: exact reused assets and their license notices.

Normalization replaces each whitespace run with one ASCII space and strips
edge whitespace. It does NOT lowercase or delete punctuation/digits. CER is
unit-cost Levenshtein distance / normalized reference character count,
including remaining spaces. DP ties prefer substitution,deletion,insertion;
this may alter the operation breakdown but not the minimum distance.
Pooled CER sums errors and reference lengths, rather than averaging percentages.

This is not a study of all documents,languages or OCR products. The English
model's confidence is not measured accuracy. No Arabic OCR or programs from
the linked companion article are tested.

## Publishing

Use Posit Connect Cloud Static Document, primary index.html,public access.
Publish robots.txt at the actual served root. Once the final sharing URL is
known, add its exact value to canonical.txt and rerun build_report.py,then
republish. Anonymous HTTP and DOM checks must verify the final surface.

The free public GitHub Action reproduces this experiment and saves its outputs
in this repository only. The engine runs locally on the Ubuntu runner; this is
not a browser OCR service. AI assistance is disclosed; no independent human
review is claimed. Original content is MIT; external font/model licenses remain
in their own folders. No secrets or personal files are used.
