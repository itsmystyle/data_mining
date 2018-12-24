#!/bin/bash

python hw5_4_preprocessing.py $2 $3

# $1/svm-train -t 2 -c 220 -v 5 $2.tr $2.model &>> $2.log
$1/svm-train -t 2 -c 220 $2.tr $2.model
# $1/svm-predict $2.tr $2.model $4 &>> $2.log
$1/svm-predict $3.te $2.model $4

# echo $1

# echo 'Without scaling' &>> $2.log

# for c in $(seq 100 2 500)
# do
# echo 'c'$c &>> $2.log
# for ((i=0;i<4;i=i+1))
# do
# $1/svm-train -t 2 -c 220 -v 10 $2 $2.model &>> $2.log
# $1/svm-train -t 2 -c 220 $2 $2.model &>> $2.log
# $1/svm-predict $2 $2.model $4 &>> $2.log
# $1/svm-predict $3 $2.model $4 &>> $2.log
# done
# done

# echo 'Scaling'
# ./$1/svm-scale -l $5 -u $6 -s range $2 > $2.scale
# ./$1/svm-scale -r range $3 > $3.scale
# echo 'Saved $2.scale/$3.scale'

# echo 'With scaling lower ' $5 ' upper ' $6 &>> $2.log
# for c in $(seq 25 1 100)
# do
# echo 'c'$c &>> $2.log
# for((i=0;i<1;i=i+1))
# do
# ./$1/svm-train -t 2 -c 100 -g 0.01 $2.scale $2'-scaled'.model &>> $2.log
# ./$1/svm-predict $3.scale $2'-scaled'.model $4'_wv_scaled_'$i.csv &>> $2.log
# done
# done



# for i in $(seq 0 1 3)
# do
# for c in $(seq 1.0 1.0 400.0)
# do
# # Calculate cross validation of each combination, 10-fold
# $1/svm-train -t $i -c $c -v 10 $2 $2.model &>> $2.log

# # Retrain model
# $1/svm-train -t $i -c $c $2 $2.model &>> $2.log

# # Calculate training accuracy
# $1/svm-predict $2 $2.model $4 &>> $2.log
# done
# done

