#!/usr/bin/env bash
#
# sync_course_to_atlas.sh — ship the rendered course videos to the portfolio site.
#
# The LLM course (this repo) is the content pipeline; the Football Atlas React
# app is the unified portfolio site that presents the videos. This script is the
# hand-off contract between them: it copies each day's final cut into the Atlas's
# public/course/ as dayN.mp4 (ascii name the web app expects) and regenerates a
# poster thumbnail. After running, rebuild + deploy the Atlas to publish:
#
#   cd "<atlas>" && ./deploy.sh
#
# Vite copies public/ into dist/, so the videos ride along in the normal deploy
# (they total ~50MB) — no separate upload step needed.
#
# Usage:
#   ./sync_course_to_atlas.sh                 # uses default Atlas path below
#   ATLAS_DIR="/path/to/football Atlas" ./sync_course_to_atlas.sh
#
set -euo pipefail

ATLAS_DIR="${ATLAS_DIR:-/Users/zouyuan/football Atlas}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/final"
DEST_DIR="$ATLAS_DIR/public/course"

POSTER_AT=2.5   # seconds — frame to grab as the poster (title card is visible)

[[ -d "$SRC_DIR" ]]   || { echo "Error: final/ not found at $SRC_DIR" >&2; exit 1; }
[[ -d "$ATLAS_DIR" ]] || { echo "Error: Atlas not found at $ATLAS_DIR" >&2; exit 1; }
mkdir -p "$DEST_DIR"

shopt -s nullglob
synced=0
for n in 1 2 3 4 5 6 7 8; do
  matches=( "$SRC_DIR"/day${n}_*.mp4 )
  if (( ${#matches[@]} == 0 )); then
    echo "  · day${n}: no final found, skipping"
    continue
  fi
  src="${matches[0]}"
  cp "$src" "$DEST_DIR/day${n}.mp4"
  ffmpeg -ss "$POSTER_AT" -i "$DEST_DIR/day${n}.mp4" -frames:v 1 -update 1 -q:v 3 \
    -y "$DEST_DIR/day${n}.jpg" -hide_banner -loglevel error
  printf "  ✓ day%d  ← %s\n" "$n" "$(basename "$src")"
  synced=$((synced + 1))
done

echo
echo "Synced $synced day(s) → $DEST_DIR"
echo "Next: cd \"$ATLAS_DIR\" && ./deploy.sh   # rebuild + publish to footballplayatlas.com"
