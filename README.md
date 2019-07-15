# HW2
## To change default implementation (TORCH / VECTORIZED / SCALAR) please change Enum var DEFAULT_IMPL at [file](neural_net/conv_net.py)

## Task 1.
For batch size = 1
And 1 epoch, checked that `diff_mse` is the same
```
Mse diff:
Conv layer: 7.853249876842248e-15,
 	Pool layer: 9.127989569713765e-15,
 	Reshape layer: 9.127989569713765e-15,
 	FC layer: 2.2586452022924086e-13
```
Please, dont use scalar implementation for learning, it causes errors, because of 
```
torch.no_grad()
```
And used only for calculation mse error.

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
