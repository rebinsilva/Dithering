all: basic block image_comparer pixel

basic: basic.c
	gcc basic.c -lm -g -o basic

block: block.c
	gcc block.c -lm -fopenmp -g -o block

image_comparer: image_comparer.c
	gcc image_comparer.c -lm -o image_comparer

pixel: pixel.cu
	nvcc pixel.cu -g -G -o pixel