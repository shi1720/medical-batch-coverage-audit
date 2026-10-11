#!/usr/bin/env sh
# Rebuild the manuscript from archived probabilities; no training or downloads.
set -eu
cd "$(dirname "$0")"
RESEARCH_PYTHON="${RESEARCH_PYTHON:-python3}"
"$RESEARCH_PYTHON" src/analyze.py
"$RESEARCH_PYTHON" src/analyze_highres.py
"$RESEARCH_PYTHON" src/additional_analysis.py
"$RESEARCH_PYTHON" src/score_sensitivity.py
"$RESEARCH_PYTHON" src/report.py
(cd paper && tectonic main.tex)
"$RESEARCH_PYTHON" src/validate_release.py
