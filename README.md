# HW2
## To change default implementation (TORCH / VECTORIZED / SCALAR) please change Enum var DEFAULT_IMPL at [file](neural_net/conv_net.py)

## Task 1.
For batch size = 1
And 1 epoch, checked that `diff_mse` is the same

## Task 2.
1. Implemented convolutional layer with im2col trick
2. Pooling layer through im2col
3. FC implemented

## Task 3.
20 epochs
1. Vectorized - \
Test set: Average loss: -0.9853, Accuracy: 9877/10000 (98.77%) \
Time: 
2. Torch - \
Test set: Average loss: -0.9853, Accuracy: 9876/10000 (98.76%)
Time:
