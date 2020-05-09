all: basic block image_comparer pixel hybrid

basic: basic.c
	gcc basic.c -lm -O3 -o basic

block: block.c
	gcc block.c -lm -fopenmp -O3 -o block

image_comparer: image_comparer.c
	gcc image_comparer.c -lm -o image_comparer

pixel: pixel.cu
	nvcc pixel.cu -o pixel
	
hybrid: hybrid.cu
	nvcc hybrid.cu -o hybrid
