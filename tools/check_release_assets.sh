#!/usr/bin/env bash
# Read asset names (one per line) on stdin. Exit 1 unless they equal the expected list.
set -euo pipefail

expected="fig1_workflow.pdf
fig2_fingerprints.pdf
fig3_phase_comparison.pdf
fig4_agreement_map.pdf
fig5_pyroxene_tima.pdf
fig6_xenotime_callout.pdf
fig7_pyroxene_refinement.pdf
fig8_lamellae.pdf
table1_params.tex
table2_phases.tex
table3_correspondence.tex
table4_concordance.tex
numbers.tex
numbers.json
acceptance-2026.md"

actual=$(sort -u)
want=$(printf '%s\n' "$expected" | sort -u)
missing=$(comm -13 <(printf '%s\n' "$actual") <(printf '%s\n' "$want"))
extra=$(comm -23 <(printf '%s\n' "$actual") <(printf '%s\n' "$want"))

if [ -n "$missing" ] || [ -n "$extra" ]; then
  [ -z "$missing" ] || { echo "Missing assets:"; printf '  %s\n' $missing; }
  [ -z "$extra" ] || { echo "Unexpected assets:"; printf '  %s\n' $extra; }
  exit 1
fi
echo "Release assets match the expected list (15 names)."
