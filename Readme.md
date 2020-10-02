# Readme

## Modulation Types and Corresponding Label

BPSK: 0
DQPSK: 1
MSK: 2
AMSSB: 3
2FSK: 4
8FSK: 5
QPSK: 6
AMDSB: 7
4FSK: 8
QAM: 9
8PSK: 10

## Changeable parameters

There are not many parameters can be changed, we can edit them directly in the files instead of `argparse`.

1. The path of training dataset and test dataset.  

   It can be modified in File `dataLoader.py, Line 18 and Line 19`

2. The GPU or CPU device to use.

   It can be modified in File `main.py, Line 16`

3. The model path and the figure path to save

   It can be modified in File `main.py, Line 46 and Line 47`

   

