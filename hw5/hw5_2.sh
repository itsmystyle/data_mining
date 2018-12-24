#!/bin/bash


# $1/svm-train -t 0 -c 0.675 -v 10 $2 $2.model &>> $2.log
$1/svm-train -t 0 -c 0.675 $2 $2.model
$1/svm-predict $3 $2.model $4 

# $1/svm-train -t 0 -c 0.675 -v 5 $2 $2.model

# echo $1

# echo 'Without scaling' &>> $2.log

# # # 0.67
# for((i=0;i<4;i=i+1))
# do
# $1/svm-train -t $i $2 $2.model
# $1/svm-predict $3 $2.model $4 &>> $2.log
# done

# echo 'Scaling' &>> $2.log
# $1/svm-scale -l $5 -u $6 -s range $2 > $2.scale
# $1/svm-scale -r range $3 > $3.scale
# echo 'Saved $2.scale/$3.scale'

# for((i=0;i<4;i=i+1))
# do
# $1/svm-train -t $i $2.scale $2'-scaled'.model
# $1/svm-predict $3.scale $2'-scaled'.model $4 &>> $2.log
# done