#!/bin/bash

echo $1

# echo 'Without scaling' &>> $2.log

# for ((i=0;i<4;i=i+1))
# do
# $1/svm-train -t $i $2 $2.model
# # $1/svm-predict $2 $2.model $4 &>> $2.log
# $1/svm-predict $3 $2.model $4 &>> $2.log
# # echo '' &>> $2.log
# done

# csv to libsvm format
python hw5_3_preprocessing.py $2 $3

$1/svm-scale -l -5 -u 5 -s range $2.tr > $2.scale
$1/svm-scale -r range $3.te > $3.scale

$1/svm-train -t 1 $2.scale $2'-scaled'.model
# $1/svm-predict $2.scale $2'-scaled'.model $4 &>> $2.log
$1/svm-predict $3.scale $2'-scaled'.model $4