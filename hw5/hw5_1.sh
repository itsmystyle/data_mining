#!/bin/bash

# echo 'Scaling'
# $1/svm-scale -l $5 -u $6 -s range $2 > $2.scale
# $1/svm-scale -r range $3 > $3.scale
# echo 'Saved $2.scale/$3.scale'

$1/svm-train -t 0 $2 $2.model
# $1/svm-predict $2 $2.model $4'_wo_scaled_'$i.csv &>> $2.log
$1/svm-predict $3 $2.model $4

# echo 'With scaling lower ' $5 ' upper ' $6 &>> $2.log

# for((i=0;i<4;i=i+1))
# do
# ./$1/svm-train -t $i $2.scale $2'-scaled'.model
# ./$1/svm-predict $3.scale $2'-scaled'.model $4'_wv_scaled_'$i.csv &>> $2.log
# done