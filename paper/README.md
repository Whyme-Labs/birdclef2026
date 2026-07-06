# BirdCLEF+ 2026 working note — LaTeX source (camera-ready)

CEUR-WS single-column working note for CLEF / LifeCLEF 2026, built with the
**official CEURART class** (`ceurart.cls` v0.6.2, 2025-10-06, from
<https://github.com/yamadharma/ceurart>), which is checked in here alongside
`main.tex`. The class pulls in the Libertinus fonts and the
`elsarticle-num-names` bibliography style; on Debian/Ubuntu these come from
`texlive-fonts-extra` + `texlive-publishers` (a full TeX Live also works).

**Build** (`pdflatex` + `bibtex`):

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Output: **`main.pdf`** (8 pages, PDF/A metadata via `pdfx`; all 9 references
resolved). This is the camera-ready PDF.

## Camera-ready revision (addressing the two accept-with-revisions reviews)

- **Official template.** Replaced the earlier `article`-class preview shim with
  the genuine `ceurart.cls` (Reviewer 2, mandatory).
- **Mandatory citations added** to `refs.bib` and cited in the introduction: the
  LifeCLEF 2026 overview (`lifeclef2026`) and the BirdCLEF+ 2026 task overview
  (`birdclef2026overview`).
- **Terminology (Reviewer 1).** The coined "measurement-gate" is now grounded in
  standard practice — a *local cross-validation split* used as a pre-submission
  gate — throughout (see §4 and §9).
- **Leak-control setup (Reviewer 2).** New §4.1 spells out how the offline split
  is constructed to avoid leakage (file-level hold-out of 13 soundscapes excluded
  from every training stage, including pseudo-label generation) and why the public
  components remain leak-optimistic on it.

The author block (`Whyme Labs`, `wmhy.tech@gmail.com`, Malaysia) and the CLEF 2026
`\conference` line (Jena, Germany, Sep 21–24, 2026) are filled. `refs.bib` carries
real metadata for all entries.
