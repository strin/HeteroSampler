#!/bin/sh
cat ../eng_ner/train | awk '{if($1 != "" && $1 != "-DOCSTART-"){ print $1" "$2"/"$4} else{print $1, $4}}' > train
cat ../eng_ner/test | awk '{if($1 != "" && $1 != "-DOCSTART-"){ print $1" "$2"/"$4} else{print $1, $4}}' > test
