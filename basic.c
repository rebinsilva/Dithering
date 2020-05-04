#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

uint8_t nearest_color(uint8_t in, uint8_t intervalLen)
{
	int temp = round(((float)in)/intervalLen);
	return temp*intervalLen;
}

void basic(unsigned char* img, unsigned char* d_img, size_t img_size, int width, int channels,uint8_t intervalLen)
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
			temp = img[i + channels] + (err*7)/16;
			img[i + channels] = (temp>255)?255:temp;
		}
		if (i + channels*(width - 1) < img_size && (i/channels) % width != 0)
		{
			temp = img[i + channels*(width - 1)] + (err*3)/16;
			img[i + channels*(width - 1)] = (temp>255)?255:temp;
		}
		if (i + channels*width < img_size)
		{
			temp = img[i + channels*width] + (err*5)/16;
			img[i + channels*width] = (temp>255)?255:temp;
		}
		if (i + channels*(width + 1) < img_size && ((i/channels) + 1) % width != 0)
		{
			temp = img[i + channels*(width + 1)] + err/16;
			img[i + channels*(width + 1)] = (temp>255)?255:temp;
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
	if (d_img == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	basic(img, d_img, img_size, width, channels, intervalLen);
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);

	stbi_image_free(img);
	free(d_img);
	return 0;
}
