#!/bin/sh
cat ../eng_ner/train | awk '{if($1 != ""){ print $1" "$2"/"$4} else{print;}}' > train
cat ../eng_ner/test | awk '{if($1 != ""){ print $1" "$2"/"$4} else{print;}}' > test
