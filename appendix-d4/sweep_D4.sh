#!/usr/bin/env bash
set -euo pipefail

echo "=== Appendix D.4 numerical sweeps ==="

BASE_OUTDIR="D-appendix-technical/sweeps"
FMT="pdf"

mkdir -p "${BASE_OUTDIR}"

run_case () {
  local TAG="$1"
  shift

  local OUTDIR="${BASE_OUTDIR}/${TAG}"
  mkdir -p "${OUTDIR}"

  echo
  echo ">>> ${TAG}"
  python chi_relaxation_validation.py \
    --outdir "${OUTDIR}" \
    --save_fmt "${FMT}" \
    "$@"

  echo "Saved to: ${OUTDIR}"
  echo "CSV: ${OUTDIR}/summary_D4.csv"
}

# --------------------------------------------------
# 0) Référence (figure officielle du papier)
# --------------------------------------------------
run_case "ref_softening_on_default"

# --------------------------------------------------
# 1) Robustesse : softening OFF
# --------------------------------------------------
run_case "robust_no_softening" --no_softening

# --------------------------------------------------
# 2) Stress test : couplage fort
# --------------------------------------------------
run_case "stress_no_softening_K0_6" --no_softening --K0 6.0

# --------------------------------------------------
# 3) Résolution (optionnel, usage interne)
# --------------------------------------------------
for N in 24 32 48; do
  run_case "res_N${N}" --N "${N}"
done

echo
echo "=== Sweep completed ==="
echo "All outputs under: ${BASE_OUTDIR}"
