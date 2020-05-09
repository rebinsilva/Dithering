#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main(int argc, char* argv[])
{
	int width, height, channels;
	unsigned char *img = stbi_load(argv[1],&width, &height, &channels, 0);
	if (img == NULL)
	{
		printf("Error in loading the image");
		exit(1);
	}
	printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
	if (channels <= 2)
	{
		printf("Already a greyscale image\n");
		exit(1);
	}

	size_t img_size = width*height*channels;
	int new_channels = channels==4?2:1;
	size_t out_size = width*height*new_channels;

	unsigned char* d_img = malloc(img_size);
	if (d_img == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}
	
	for (unsigned char *p = img, *pd = d_img; p!=img+img_size; p += channels, pd += new_channels)
	{
		*pd = (unsigned char)((*p + *(p+1) + *(p+2))/3);
		if(channels == 4)
		{
			*(pd + 1) = *(p + 3);
		}
	}
	stbi_write_png(argv[2], width, height, new_channels, d_img, width*new_channels);

	stbi_image_free(img);
	free(d_img);
	return 0;
}
