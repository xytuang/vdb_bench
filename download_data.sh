#!/bin/bash

mkdir -p /mydata/SPTAG

sudo apt install git-lfs -y
git lfs install


GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/microsoft/SPTAG.git /mydata/SPTAG

cd /mydata/SPTAG

for i in {1..15}; do
    git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_$i.bin"
done

git lfs pull --include "datasets/SPACEV1B/query.bin"
git lfs pull --include "datasets/SPACEV1B/truth.bin"

cd datasets/SPACEV1B/vectors.bin/
mv vectors_1.bin vectors_merged.bin
for i in {2..15}; do
	cat vectors_$i.bin >> vectors_merged.bin
	rm vectors_$i.bin
done
