#!/bin/bash
# Offload finished self-play archive files from the engine container to Google Drive.
#
# The server appends every self-play game to training_archive/selfplay_YYYYMMDD.jsonl.gz
# (and the one-off legacy export lives there too). Files from past days are immutable.
# For each such file: copy it out of the container, upload with rclone, verify size +
# checksum on Drive, and only then (with --delete-local) remove the container copy.
# Today's file is never touched. The games also remain in training.db.
#
# Usage: scripts/archive_training_to_drive.sh [--delete-local] [--dest gdrive:knightball/training_archive]
set -euo pipefail
CONTAINER=${CONTAINER:-knightball-engine}
SRC=/app/server/data/training_archive
DEST=gdrive:knightball/training_archive
DELETE=0
while [ $# -gt 0 ]; do
  case $1 in
    --delete-local) DELETE=1 ;;
    --dest) DEST=$2; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
  shift
done
TODAY=$(date -u +%Y%m%d)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

for f in $(docker exec "$CONTAINER" sh -c "ls $SRC 2>/dev/null"); do
  case $f in *"$TODAY"*) echo "skip $f (today, still being written)"; continue ;; esac
  case $f in *.jsonl.gz) ;; *) continue ;; esac
  docker cp "$CONTAINER:$SRC/$f" "$STAGE/$f"
  gzip -t "$STAGE/$f"
  rclone copy "$STAGE/$f" "$DEST" --checksum
  if rclone check "$STAGE" "$DEST" --include "$f" --one-way >/dev/null 2>&1; then
    echo "uploaded + verified $f ($(du -h "$STAGE/$f" | cut -f1))"
    if [ "$DELETE" = 1 ]; then
      docker exec "$CONTAINER" rm "$SRC/$f"
      echo "  removed local copy"
    fi
  else
    echo "VERIFY FAILED for $f — local copy kept" >&2
  fi
  rm -f "$STAGE/$f"
done
