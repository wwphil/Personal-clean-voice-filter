#!/usr/bin/env bash
# Re-create vendor/ at the pinned commits (see VENDOR.md).
set -euo pipefail
cd "$(dirname "$0")/.."

clone() { # url branch commit name
  local url=$1 branch=$2 commit=$3 name=$4
  if [ -d "vendor/$name/.git" ]; then
    echo "== $name: present, checking pin"
    actual=$(git -C "vendor/$name" rev-parse HEAD)
    if [ "$actual" != "$commit" ]; then
      echo "   pinned $commit, found $actual -> re-cloning"
      rm -rf "vendor/$name"
      git clone --quiet --branch "$branch" "$url" "vendor/$name"
      git -C "vendor/$name" checkout --quiet "$commit"
    fi
    return
  fi
  echo "== $name: cloning $branch @ $commit"
  git clone --quiet --branch "$branch" "$url" "vendor/$name"
  git -C "vendor/$name" checkout --quiet "$commit"
}

clone https://github.com/Rikorose/DeepFilterNet main \
      7b14b895fe7fc42cab72b4a4c1e49c982bd6404b DeepFilterNet
clone https://github.com/wenet-e2e/WeSpeaker master \
      9fecd6cb4f47475d01761d87c826298dff4ef18c WeSpeaker
clone https://github.com/wenet-e2e/wesep wesep_2024 \
      4c899c49c300a83f9cfd0d06a4acde7894abae0c wesep

echo "vendor/ ready"
