#!/usr/bin/env bash

#declare -a embeddings=("lipschitz" "lmds" "fastmap" "dissimilarity" "resampling")
declare -a embeddings=("lmds")
#declare -a tracks=("100307" "102311" "109123")
declare -a tracks=( "109123")
for t in "${tracks[@]}"
    do
        for j in "${embeddings[@]}"
            do
                for i in `seq 10 20`;
                        do
                                python experiment.py $i $j $t
                        done
            done
    done