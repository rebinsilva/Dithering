#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main(int argc, char* argv[])
{
	int width1, height1, channels1, width2, height2, channels2;
	unsigned char *img1 = stbi_load(argv[1],&width1, &height1, &channels1, 0);
    unsigned char *img2 = stbi_load(argv[2],&width2, &height2, &channels2, 0);

	if (img1 == NULL)
	{
		printf("Error in loading the image 1\n");
		exit(1);
	}
	if (img2 == NULL)
	{
		printf("Error in loading the image 2\n");
		exit(1);
	}

	if (channels1 > 2 || channels2 > 2)
	{
		printf("Not a grayscale image\n");
		exit(1);
	}

    if (width1 != width2 || height1 != height2 || channels1 != channels2)
    {
        printf("Images with different resolution\n");
		exit(1);
    }

	size_t img_size = width1*height1*channels1;

	unsigned char* d_img = calloc(img_size, sizeof(unsigned char));
	for(int i=0; i<img_size; i += channels1)
    {
        if(img1[i] == img2[i])
            d_img[i] = 255;
        else 
            d_img[i] = 0;
        d_img[i+1] = (img1[i+1] + img2[i+1])/2;
    }
	stbi_write_png(argv[3], width1, height1, channels1, d_img, width1*channels1);

	return 0;
}
