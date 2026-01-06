#!/bin/bash
set -euo pipefail

SOURCE_VERSION_FILE="setup.py"
VERSION_FILES=(
  "setup.py"
  "arcgispro_ai/__init__.py"
  "meta.yaml"
)

RAW_LINE=$(grep -m1 "version" "$SOURCE_VERSION_FILE")
CURRENT_VERSION=$(echo "$RAW_LINE" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

if [[ -z "$CURRENT_VERSION" ]]; then
  echo "Could not find version in $SOURCE_VERSION_FILE"
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"

echo "Current version: $CURRENT_VERSION"
echo "Bumping to: $NEW_VERSION"

python - <<'PY' "$CURRENT_VERSION" "$NEW_VERSION" "${VERSION_FILES[@]}"
import pathlib
import sys

current, new, *files = sys.argv[1:]

for file in files:
    path = pathlib.Path(file)
    text = path.read_text()
    if current not in text:
        raise SystemExit(f"{current} not found in {file}")
    path.write_text(text.replace(current, new))
    print(f"Updated {file}")
PY

git add "${VERSION_FILES[@]}"
git commit -m "Bump version to $NEW_VERSION"

TAG="v$NEW_VERSION"
git tag -a "$TAG" -m "Release $NEW_VERSION"
git push origin main
git push origin "$TAG"

REPO_URL=$(git config --get remote.origin.url | sed 's/.*github.com[/:]\(.*\)\.git/\1/')
URL="https://github.com/$REPO_URL/releases/tag/$TAG"

( open "$URL" || xdg-open "$URL" ) 2>/dev/null || echo "View the release at: $URL"
