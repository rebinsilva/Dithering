#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

uint8_t nearest_color(int in, uint8_t intervalLen)
{
	in = (in > 255)?255:in;
	int temp = round(((float)in)/intervalLen);
	return temp*intervalLen;
}

void basic(int* img, unsigned char* d_img, size_t img_size, int width, int channels,uint8_t intervalLen)
{
	int err = 0, temp = 0;
	for (unsigned int i = 0; i<img_size; i += channels)
	{
		d_img[i] = nearest_color(img[i], intervalLen);
		if (channels == 2)
		{
			d_img[i+1] = img[i+1];
		}
		err = img[i] - d_img[i];
		if (i + channels < img_size)
		{
			if (((i/channels) + 1) % width != 0)
			{
				img[i + channels] += (err*7)/16;
			}
			if (i + channels*(width - 1) < img_size)
			{
				if ((i/channels) % width != 0)
				{
					img[i + channels*(width - 1)] += (err*3)/16;
				}
				if (i + channels*width < img_size)
				{
					img[i + channels*width] += (err*5)/16;
					if (i + channels*(width + 1) < img_size && ((i/channels) + 1) % width != 0)
					{
						img[i + channels*(width + 1)] += err/16;
					}
				}
			}
		}
	}
	return;
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

	unsigned char* d_img = calloc(img_size, sizeof(unsigned char));
	int* pre = calloc(img_size, sizeof(int));
	if (d_img == NULL || pre == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	for(int i=0; i < img_size; i++)
	{
		pre[i] = img[i];
	}
	
	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);

	basic(pre, d_img, img_size, width, channels, intervalLen);

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec) * 1000;
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000.0;
	printf("Time taken in milliseconds: %f\n", elapsed);
	
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);

	stbi_image_free(img);
	free(d_img);
	return 0;
}
