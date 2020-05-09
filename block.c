#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

typedef struct PrimalBlock
{
	int row;
	int col;
}PrimalBlock;

uint8_t nearest_color(int in, uint8_t intervalLen)
{
	in = (in > 255)?255:in;
	int temp = round(((float)in)/intervalLen);
	return temp*intervalLen;
}

PrimalBlock pb_finder(int M, int N, int a, int b, int itr)
{
	PrimalBlock ans;
	if (itr > 0 && itr <= ceil((float)N/b)) 
	{
		ans.row = 1;
		ans.col = itr;
	}
	else
	{
		ans.row = 1 + ceil((itr - ceil((float)N/b))/2);
		ans.col = ceil((float)N/b) - (int)(itr - ceil((float)N/b))%2;
	}
	return ans;
}

void dither(int height, int width, int channels, int img[height][width*channels], unsigned char* d_img, uint8_t intervalLen, int row, int col)
{
	int i=row,j=col;
	d_img[i*width*channels + j] = nearest_color(img[i][j], intervalLen);
	if (channels == 2)
	{
		d_img[i*width*channels + j + 1] = img[i][j +1];
	}
	int err = img[i][j] - d_img[i*width*channels + j];
	if (j+channels < width*channels)
	{
		img[i][j+channels] += (err*7)/16;
	}
	if (i + 1 < height)
	{
		if (j != 0)
		{
			img[i+1][j-channels] += (err*3)/16;
		}
		img[i+1][j] += (err*5)/16;
		if (j + channels < width*channels)
		{
			img[i+1][j+channels] += err/16;
		}
	}
}

void ditherblock(int height, int width, int channels, int img[height][width], unsigned char* d_img, uint8_t intervalLen, int row, int col, int a, int b)
{
	row = a*row;
	col = b*col;
	for (int i=row;i<row+a && i<height;i++)
	{
		for (int j=(col==0)?col:(col+a-i+row); j<col + b + a - i + row && j<width; j++)
		{
			dither(height, width, channels, img, d_img, intervalLen, i, j*channels);
		}
	}
}

void block(int height, int width, int channels, int img[height][width], unsigned char* d_img, uint8_t intervalLen, int a, int b)
{
	int row = 0;
	float col = 0;
	for (int i=1; i <= 2*ceil((height-a)/(float)a) + ceil(width/(float)b); i++)
	{
		PrimalBlock pb = pb_finder(height, width, a, b, i);
		row = pb.row-1;
		col = pb.col-1;
		//printf("%d %d\n", row, col);
		int mini = fmin(height - row -1, col/2);
		omp_set_num_threads(8);
		#pragma omp parallel for
		for(int k=0; k <= mini; k++)
		{
			ditherblock(height, width, channels, img, d_img, intervalLen, row+k, (col-2*k)*channels, a, b);
		}
	}
}

int main(int argc, char* argv[])
{
	int width, height, channels;
	unsigned char *img = stbi_load(argv[2],&width, &height, &channels, 0);
	uint8_t intervalLen = 255/(strtol(argv[1], NULL, 10)-1);

	if (img == NULL)
	{
		printf("Error in loading the image");
		exit(1);
	}
	printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
	if (channels > 2)
	{
		printf("Not a grayscale image\n");
		exit(1);
	}

	size_t img_size = width*height*channels;
	int block_size = (width*height)/16;
	int a = sqrt(block_size), b = a;

	unsigned char* d_img = calloc(img_size, sizeof(unsigned char));
	int pre[height][width*channels];
	if (d_img == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	for(int i=0; i < height; i++)
	{
		for (int j=0; j < width*channels; j++)
		{
			pre[i][j] = img[i*width*channels + j];
		}
	}
	
	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);

	block(height, width, channels, pre, d_img, intervalLen, a, b);

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec) * 1000;
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000.0;
	printf("Time taken in milliseconds: %f\n", elapsed);
	
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);

	return 0;
}
