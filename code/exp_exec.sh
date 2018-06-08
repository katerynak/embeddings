#!/usr/bin/env bash

#declare -a tracks=("100307" "102311" "109123")
declare -a embeddings=("lmds", "fastmap", "dissimilarity", "lipschitz", "resampling")
declare -a tracks=( "109123")
for t in "${tracks[@]}"
    do
        for e in "${embeddings[@]}"
            do
                python experiment.py $e $t
            done
    done