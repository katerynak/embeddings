#!/usr/bin/env bash

#declare -a tracks=("100307" "102311" "109123")
declare -a tracks=( "109123")
for t in "${tracks[@]}"
    do
        python experiment.py $t
    done