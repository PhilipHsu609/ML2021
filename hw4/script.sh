#!/bin/bash

train="./data/hw4_train.dat"
test="./data/hw4_test.dat"

train_transform="./tmp/train_transform.dat"
test_transform="./tmp/test_transform.dat"

# Temporary directory
mkdir ./tmp

# Feature transform
./transform.out $train $train_transform
./transform.out $test $test_transform

# Q12
echo -e "\n------- Question 12 -------\n"

for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    ./train -s 0 -c $C -e 0.000001 -q $train_transform ./tmp/model
    echo -e "  Out-sample: " $(./predict -b 1 $test_transform ./tmp/model ./tmp/label) "\n"
done

# Q13
echo -e "\n------- Question 13 -------\n"

for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    ./train -s 0 -c $C -e 0.000001 -q $train_transform ./tmp/model
    echo -e "  In-sample: " $(./predict -b 1 $train_transform ./tmp/model ./tmp/label) "\n"
done

# Q14
echo -e "\n------- Question 14 -------\n"

# Split $train_transform into D_train(Q14_00.dat) and D_val(Q14_01.dat)
split -l 120 -d --additional-suffix=.dat $train_transform ./tmp/Q14_

for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    ./train -s 0 -c $C -e 0.000001 -q ./tmp/Q14_00.dat ./tmp/model
    echo -e "  Val-sample: " $(./predict -b 1 ./tmp/Q14_01.dat ./tmp/model ./tmp/label)
    echo -e "  Out-sample: " $(./predict -b 1 $test_transform ./tmp/model ./tmp/label) "\n"
done

# Q15
echo -e "\n------- Question 15 -------\n"

i=2
lambda=$(echo "10 ^ $i" | bc -l)
C=$(echo "1 / $lambda / 2" | bc -l)
echo "\log_10(\lambda) = $i"

./train -s 0 -c $C -e 0.000001 -q $train_transform ./tmp/model
./predict -b 1 $test_transform ./tmp/model ./tmp/label

# Q16
echo  -e "\n------- Question 16 -------\n"

# Split $train_transform into 5 parts, Q16_00.dat to Q16_04.dat
split -l 40 -d --additional-suffix=.dat $train_transform ./tmp/Q16_

# Taking 4 parts for training.
#   Q16_train_$i.dat for training, Q16_0$i.dat for validation.
cd tmp
for i in {0..4}; do
    cat $(ls | grep "Q16_0[^$i].dat") > Q16_train_$i.dat
done
cd ../

for i in -4 -2 0 2 4; do
    lambda=$(echo "10 ^ $i" | bc -l)
    C=$(echo "1 / $lambda / 2" | bc -l)
    echo "\log_10(\lambda) = $i"

    avg=0
    for j in {0..4}; do
        ./train -s 0 -c $C -e 0.000001 -q ./tmp/Q16_train_$j.dat ./tmp/model
        acc=$(./predict -b 1 ./tmp/Q16_0$j.dat ./tmp/model ./tmp/label | grep -oP "\d+\.?\d*(?=%)")
        avg=$(echo "$avg + (1 - $acc/100)" | bc -l)
    done
    echo -e "  E_cv = " $(echo "$avg / 5" | bc -l) "\n"
done

# Cleaning
rm -rf ./tmp