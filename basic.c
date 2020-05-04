#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

uint8_t nearest_color(uint8_t in, uint8_t intervalLen)
{
	in = (in>255)?255:in;
	int temp = round(((float)in)/intervalLen);
	return temp*intervalLen;
}

void basic(unsigned char* img, unsigned char* d_img, size_t img_size, int width, int channels,uint8_t intervalLen)
{
	for (unsigned char *p = img, *pd = d_img; p!=img+img_size; p += channels, pd += channels)
	{
		*pd = nearest_color(*p, intervalLen);
		if(channels == 2)
		{
			*(pd + 1) = *(p + 1);
		}
		int err = *p - *pd;
		if (p + channels < img + img_size)
			*(p + channels) += (err*7)/16;
		if (p + channels*(width-1) < img + img_size)
			*(p + channels*(width - 1)) += (err*3)/16;
		if (p + channels*width < img + img_size)
			*(p + channels*width) += (err*5)/16;
		if (p + channels*(width + 1) < img + img_size)
			*(p + channels*(width + 1)) += (err)/16;
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
