#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

typedef struct PrimalBlock
{
	int row;
	int col;
}PrimalBlock;

__host__ __device__ uint8_t nearest_color(float in, uint8_t intervalLen)
{
	in = fminf(in, 255);
	in = fmaxf(in, 0);
	int temp = round(((float)in)/intervalLen);
	return (uint8_t)temp*intervalLen;
}

PrimalBlock pb_finder(int M, int N, int itr)
{
	PrimalBlock ans;
	if (itr <= N) 
	{
		ans.row = 1;
		ans.col = itr;
	}
	else
	{
		ans.row = 1 + ceil((itr - (float)N)/2);
		ans.col = N - (itr - N)%2;
	}
	return ans;
}

void dither_cpu(bool right[], int pri, int intervalLen, float* in, unsigned char* out, int size[], int tid)
{
	out[pri+tid] = nearest_color(in[pri+tid], intervalLen);
	float err = in[pri+tid] - out[pri+tid];
	if (right[0])
	{
		if (tid !=0)
		{
			in[pri + size[0] + tid - 1] += (err*7)/16;
			if (tid - 1 < size[3])
				in[pri + size[0] + size[1] + size[2] + tid - 1] += err/16;
		}
		if (tid < size[1])
			in[pri + size[0] + tid] += (err*3)/16;
		if (tid < size[2])
			in[pri + size[0] + size[1] + tid] += (err*5)/16;
	}
	else if (right[1])
	{
		in[pri + size[0] + tid] += (err*7)/16;
		if (tid + 1 < size[1])
			in[pri + size[0] + tid + 1] += (err*3)/16;
		if (tid < size[2])
			in[pri + size[0] + size[1] + tid] += (err*5)/16;
		if (tid < size[3])
			in[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else if (right[2])
	{
		in[pri + size[0] + tid] += (err*7)/16;
		if (tid + 1 < size[1])
			in[pri + size[0] + tid + 1] += (err*3)/16;
		if (tid + 1 < size[2])
			in[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid < size[3])
			in[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else
	{
		in[pri + size[0] + tid] += (err*7)/16;
		if (tid + 1 < size[1])
			in[pri + size[0] + tid + 1] += (err*3)/16;
		if (tid + 1 < size[2])
			in[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid + 1 < size[3])
			in[pri + size[0] + size[1] + size[2] + tid + 1] += err/16;
	}
	return;
}

__global__ void dither(bool right[], int pri, int intervalLen, float* in, unsigned char* out, int size[])
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= size[0])
	{
		return;
	}
	out[pri+tid] = nearest_color(in[pri+tid], intervalLen);
	float err = in[pri+tid] - out[pri+tid];
	if (right[0])
	{
		if (tid !=0)
		{
			atomicAdd(&in[pri + size[0] + tid - 1], (err*7)/16);
			if (tid - 1 < size[3])
				in[pri + size[0] + size[1] + size[2] + tid - 1] += err/16;
		}
		if (tid < size[1])
			atomicAdd(&in[pri + size[0] + tid], (err*3)/16);
		if (tid < size[2])
			in[pri + size[0] + size[1] + tid] += (err*5)/16;
	}
	else if (right[1])
	{
		atomicAdd(&in[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&in[pri + size[0] + tid + 1], (err*3)/16);
		if (tid < size[2])
			in[pri + size[0] + size[1] + tid] += (err*5)/16;
		if (tid < size[3])
			in[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else if (right[2])
	{
		atomicAdd(&in[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&in[pri + size[0] + tid + 1], (err*3)/16);
		if (tid + 1 < size[2])
			in[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid < size[3])
			in[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else
	{
		atomicAdd(&in[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&in[pri + size[0] + tid + 1], (err*3)/16);
		if (tid + 1 < size[2])
			in[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid + 1 < size[3])
			in[pri + size[0] + size[1] + size[2] + tid + 1] += err/16;
	}
	return;
}

void ditherimage(int height, int width, int intervalLen, float* in, unsigned char* out, int primals[])
{
	int size[4];
	bool right[3];
	int* g_size;
	bool* g_right;
	bool isGPU = false;
	float *in_cpu = (float*)malloc(width*height*sizeof(float));
	unsigned char* out_cpu = (unsigned char*)malloc(width*height*sizeof(unsigned char));
	cudaMemcpy(in_cpu, in, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(out_cpu, out, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMalloc(&g_size, 4*sizeof(int));
	cudaMalloc(&g_right, 4*sizeof(bool));
	for (int i=1; i <= 2*(height-1) + width; i++)
	{
		PrimalBlock pb = pb_finder(height, width, i);
		for(int j=0; j< 4; j++)
		{
			size[j] = primals[i+j] - primals[i+j-1];
		}
		right[0] = pb.col == width;
		right[1] = pb.col >= width - 1;
		right[2] = pb.col >= width - 2;
		cudaMemcpy(g_size, size, 4*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(g_right, right, 3*sizeof(bool), cudaMemcpyHostToDevice);
		if(size[0] > 100)
		{
			if(!isGPU)
			{
				cudaMemcpy(in, in_cpu, width*height*sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(out, out_cpu, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
				isGPU = true;
			}
			if(size[0] < 1024)
			{
				dither<<<1,size[0]>>>(g_right, primals[i-1], intervalLen, in, out, g_size);
			}
			else
			{
				dither<<<ceil(((float)size[0])/1024),1024>>>(g_right, primals[i-1], intervalLen, in, out, g_size);
			}
		}
		else
		{
			if(isGPU)
			{
				cudaMemcpy(in_cpu, in, width*height*sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(out_cpu, out, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
				isGPU = false;
			}
		
			for(int j=0; j<size[0]; j++)
			{
				dither_cpu(right, primals[i-1], intervalLen, in_cpu, out_cpu, size, j);
			}
		}
	}

	cudaMemcpy(in, in_cpu, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(out, out_cpu, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
}

void reorder(int height, int width, int channels, unsigned char** pre, float out[], int primals[])
{
	int counter = 0;
	int row =0;
	int col = 0;
	for (int i=1; i <= 2*(height-1) + width; i++)
	{
		PrimalBlock pb = pb_finder(height, width, i);
		row = pb.row-1;
		col = pb.col-1;
		int mini = fmin(height - row -1, ((float)col)/(2));
		primals[i-1] = counter;
		for(int k=0; k <= mini; k++)
		{
			out[counter] = pre[row+k][(col-2*k)*channels];
			counter ++;
		}
	}

}

void order(int height, int width, int channels, unsigned char in[], unsigned char* img, unsigned char* post)
{
	int counter = 0;
	int row =0;
	int col = 0;
	for (int i=1; i <= 2*(height-1) + width; i++)
	{
		PrimalBlock pb = pb_finder(height, width, i);
		row = pb.row-1;
		col = pb.col-1;
		int mini = fmin(height - row -1, ((float)col)/2);
		for(int k=0; k <= mini; k++)
		{
			post[((row+k)*width+col-2*k)*channels] = in[counter];
			if (channels == 2)
				post[((row+k)*width+col-2*k)*channels + 1 ] = img[((row+k)*width+col-2*k)*channels + 1];
			counter ++;
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

	int* primals = (int*)malloc((width+2*height+2)*sizeof(int));
	unsigned char* d_img = (unsigned char*)calloc(img_size, sizeof(unsigned char));
	unsigned char* pre[height];
	float* reordered = (float*)malloc(width*height*sizeof(float));
	unsigned char* dithered = (unsigned char*)malloc(width*height*sizeof(unsigned char));
	float *g_reordered;
	unsigned char* g_dithered;
	if (d_img == NULL || primals == NULL || reordered == NULL || dithered == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	for(int i=0; i<4; i++)
		primals[width+2*(height-1) + i]=width*height;
	for(int i=0; i < height; i++)
	{
		pre[i] = img + i*width*channels;
	}
	reorder(height, width, channels, pre, reordered, primals);
	cudaMalloc(&g_dithered, width*height*sizeof(unsigned char));
	cudaMalloc(&g_reordered, width*height*sizeof(float));

	cudaMemcpy(g_reordered, reordered, width*height*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	ditherimage(height, width, intervalLen, g_reordered, g_dithered, primals);
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken in milliseconds: %f\n",milliseconds);
	
	cudaMemcpy(dithered, g_dithered, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	order(height, width, channels, dithered, img, d_img);
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);
	free(d_img);
	cudaFree(g_dithered);
	cudaFree(g_reordered);

	return 0;
}
