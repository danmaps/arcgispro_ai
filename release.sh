#!/bin/bash
set -e

SETUP_PATH="setup.py"

# Extract version line: version="0.0.7" or version='0.0.7'
RAW_LINE=$(grep version= "$SETUP_PATH")
CURRENT_VERSION=$(echo "$RAW_LINE" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

if [[ -z "$CURRENT_VERSION" ]]; then
  echo "Could not find version in $SETUP_PATH"
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"

echo "Current version: $CURRENT_VERSION"
echo "Bumping to: $NEW_VERSION"

# Replace version string in setup.py
sed -i.bak "s/$CURRENT_VERSION/$NEW_VERSION/" "$SETUP_PATH"
rm "$SETUP_PATH.bak"

# Commit version bump
git add "$SETUP_PATH"
git commit -m "Bump version to $NEW_VERSION"

# Tag and push
TAG="v$NEW_VERSION"
git tag -a "$TAG" -m "Release $NEW_VERSION"
git push origin main
git push origin "$TAG"

# Open GitHub release page (optional)
REPO_URL=$(git config --get remote.origin.url | sed 's/.*github.com[/:]\(.*\)\.git/\1/')
URL="https://github.com/$REPO_URL/releases/tag/$TAG"

( open "$URL" || xdg-open "$URL" ) 2>/dev/null || echo "View the release at: $URL"
