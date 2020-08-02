# Image Dithering
Error diffusion dithering using Floyd–Steinberg dithering algorithm is implemented in this repository. The algorithm is implemented using CPU only and CPU-GPU hybri methods. This repository is an re-implementation of the paper [Hybrid Implementation of Error Diffusion Dithering]. 

## Codes

### basic.c
This is a sequential implementation of Floyd-Steinberg dithering algorithm in CPU.

## block.c
This is a parallel implementation of dithering algorithm using openmp in CPU.

## pixel.cu
This is a parallel implementation of dithering algorithm using CPU-GPU handover algorithm.

## hybrid.cu
This is a parallel implementation of dithering algorithm using CPU-GPU hybrid algorithm.


[Hybrid Implementation of Error Diffusion Dithering]:http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/papers/Aditya2011Hybrid.pdf
