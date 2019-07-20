# Intention of the code

The most efficient implementations of CUDA sgemm (float32 Matrix x Matrix), such as cublas, uses hand-tuned sass code.

However, sass tuning is painful, and binary code is inflexible.

In this code, I'm trying to optimize the g_sgemm kernel using CUDA C only. 

The aim is to achieve a performance like [cutlass](https://github.com/NVIDIA/cutlass) (without wmma), using a very small amount of code.

# License

This code is just some practise. Please consider it public-domain.
There's no restriction of use, and absolutely no warranty.

