# WLASL Complete (Curated, Split-Balanced)

## What this is
A re-split, download-verified version of the WLASL (Word-Level American
Sign Language) dataset by Li & Li (WACV 2020), rebuilt after accounting
for real-world link rot in the original source videos.

- **1910 classes** retained (out of the original 2,000), each
  guaranteed to have at least one example in train, validation, AND test.
- **11724 total videos** (train: 7696, val: 2014, test: 2014).
- Classes dropped: 90, each because fewer than 4 of their source videos
  are still downloadable today. See `dropped_classes.txt` for the full list.
- Splits were reassigned per-class using a fixed random seed for
  reproducibility (~70/15/15 target ratio, minimum 1 example per split).
  See `nslt_custom.json` (flat video_id -> {subset, action} format) and
  `WLASL_custom_split.json` (full nested metadata, same shape as the
  original `WLASL_v0.3.json`).

## Why re-split instead of using the original WLASL split labels
The original 2020 split assignments predate significant link rot in the
source videos (many were scraped from YouTube and ASL instructional
sites). After filtering to only currently-downloadable videos, using the
original split labels leaves a large fraction of classes with zero
validation or zero test examples, making per-class accuracy reporting
unreliable. This version re-splits from what is actually available so
every retained class can be fairly trained and evaluated.

## License and attribution
This dataset is derived from WLASL, released under the **Computational
Use of Data Agreement (C-UDA-1.0)**. It is intended for **academic and
computational research use only — no commercial use permitted**.

Please cite the original authors:
> Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level Deep Sign
> Language Recognition from Video: A New Large-scale Dataset and Methods
> Comparison. WACV 2020.

Original dataset: https://dxli94.github.io/WLASL/
