#!/bin/bash

train="hw4_train.dat"
test="hw4_test.dat"

train_transform="hw4_train_transform.dat"
test_transform="hw4_test_transform.dat"

./hw4.out $train $train_transform
./hw4.out $test $test_transform

declare -a q=("Q12" "Q13" "Q14" "Q15" "Q16")

for i in "${q[@]}"; do
    if [[ ! -e $i ]]; then
        mkdir $i
    fi
done

# Q12
echo "------- Question 12 -------"
for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    ./train -s 0 -c $C -e 0.000001 -q $train_transform "Q12/${i}_model"
    ./predict -b 1 $test_transform "Q12/${i}_model" "Q12/${i}_label"
    echo
done

# Q13
echo "------- Question 13 -------"
for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    ./train -s 0 -c $C -e 0.000001 -q $train_transform "Q13/${i}_model"
    ./predict -b 1 $train_transform "Q13/${i}_model" "Q13/${i}_label"
    echo
done

# Q14
echo "------- Question 14 -------"

# Q15
echo "------- Question 15 -------"

# Q16
echo "------- Question 16 -------"