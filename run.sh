#!/bin/bash

# Check if both file names are provided as arguments

testFile="$1"
predictionFile="$2"

echo "Training model and testing on the Test File, and then predicting into the Prediction File:"
echo "Test File: $testFile"
echo "Prediction File: $predictionFile"

python model.py "$testFile" "$predictionFile"