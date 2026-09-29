# Sources, licenses, and submission assets

## Project

- Author and owner: Atharva Parande
- Source-code license: MIT; see `LICENSE`
- No external pretrained model, hosted inference API, paid service, music, icon pack, or stock
  footage is required by the submitted system.

## Primary scientific data

- Afting C et al. *A deep learning-based computational pipeline predicts developmental outcome in
  retinal organoids*. PLOS Biology 24(1):e3003597 (2026).
  DOI: `10.1371/journal.pbio.3003597`.
- Published source table: Zenodo record `18198347`, `Extended_Data_2.csv`.
- Source license: CC BY 4.0; attribution is required.
- The 400,059,241-byte table is not redistributed in Git. The downloader pins its filename, record,
  size, MD5, and SHA-256 and verifies it before use.

## Software dependencies

The runtime uses Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Plotly, Streamlit, and
Joblib. Development verification uses pytest and Ruff. Exact resolved versions and package metadata
are recorded in `uv.lock`; the project does not copy dependency source code into the repository.

## Comparative method

The CBES-style baseline cites Liu et al., *Confidence-bound early stopping of experiments with
sequential calibration*, Chemical Engineering Research and Design 230:756-766 (2026),
DOI `10.1016/j.cherd.2026.05.013`. Its public reference implementation is cited in
`docs/phase2/CBES_ADAPTATION.md`. The repository implementation is an independently documented
binary-endpoint adaptation.

## Figures, demo, and video

All submitted plots are generated from repository evidence. Figure-to-table mappings are in
`artifacts/final/figure_sources.csv`. The Streamlit interface uses no third-party visual assets.
The recording script requires only screen capture of the project and the author's narration. If
music, fonts, icons, or external media are added during recording, their licenses must be recorded
here before submission; none are currently planned.
