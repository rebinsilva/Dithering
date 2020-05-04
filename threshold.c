#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main(int argc, char* argv[])
{
	int width, height, channels;
	unsigned char *img = stbi_load(argv[2],&width, &height, &channels, 0);
	uint8_t div = 256/strtol(argv[1], NULL, 10);
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

	unsigned char* d_img = malloc(img_size);
	if (d_img == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	for (unsigned char *p = img, *pd = d_img; p!=img+img_size; p += channels, pd += channels)
	{
		uint8_t temp = (*p)/div;
		*pd = (uint8_t)(temp*div);
		if(channels == 2)
		{
			*(pd + 1) = *(p + 1);
		}
	}
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);

	stbi_image_free(img);
	free(d_img);
	return 0;
}
