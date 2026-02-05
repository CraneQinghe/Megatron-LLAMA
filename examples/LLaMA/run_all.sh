#!/bin/bash

# Iterate over dp values (data parallelism)
for dp in 1 2 4 8
do
    # Iterate over tp values (tensor parallelism)
    for (( tp = 8 / dp; tp >= 1; tp/=2 ))
    do
        # Calculate pp (pipeline parallelism)
        pp=$((8 / dp / tp))

        # Print the current combination
        echo "Running with dp=$dp, tp=$tp, pp=$pp"

        # Run the LLaMA_20B.sh script with the calculated dp, tp, and pp values
        bash LLaMA_13_standalone.sh $tp $pp $dp 
    done
done