#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int cpu_width = 80;
typedef struct PrimalBlock
{
	int row;
	int col;
}PrimalBlock;

__host__ __device__ uint8_t nearest_color(float in, uint8_t intervalLen)
{
	in = fminf(in, 255);
	in = fmaxf(in, 0);
	return round(in/intervalLen)*intervalLen;
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

__global__ void dither(bool right[], int pri, int intervalLen, int col, int cpu_width, float* in, unsigned char* out, int size[], float* zero_memory)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= size[0] || col - 2*tid < cpu_width)
	{
		return;
	}
	float *r = in, *br = in, *b = in, *bl = in;
	if (col - 2*tid == cpu_width + 1)
	{
		bl = zero_memory;
	}
	if (col - 2*tid == cpu_width)
	{
		bl = zero_memory;
		b = zero_memory;
		in = zero_memory;
	}
	out[pri+tid] = nearest_color(in[pri+tid], intervalLen);
	float err = in[pri+tid] - out[pri+tid];
	if (right[0])
	{
		if (tid !=0)
		{
			atomicAdd(&r[pri + size[0] + tid - 1], (err*7)/16);
			if (tid - 1 < size[3])
				br[pri + size[0] + size[1] + size[2] + tid - 1] += err/16;
		}
		if (tid < size[1])
			atomicAdd(&bl[pri + size[0] + tid], (err*3)/16);
		if (tid < size[2])
			b[pri + size[0] + size[1] + tid] += (err*5)/16;
	}
	else if (right[1])
	{
		atomicAdd(&r[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&bl[pri + size[0] + tid + 1], (err*3)/16);
		if (tid < size[2])
			b[pri + size[0] + size[1] + tid] += (err*5)/16;
		if (tid < size[3])
			br[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else if (right[2])
	{
		atomicAdd(&r[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&bl[pri + size[0] + tid + 1], (err*3)/16);
		if (tid + 1 < size[2])
			b[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid < size[3])
			br[pri + size[0] + size[1] + size[2] + tid] += err/16;
	}
	else
	{
		atomicAdd(&r[pri + size[0] + tid], (err*7)/16);
		if (tid + 1 < size[1])
			atomicAdd(&bl[pri + size[0] + tid + 1], (err*3)/16);
		if (tid + 1 < size[2])
			b[pri + size[0] + size[1] + tid + 1] += (err*5)/16;
		if (tid + 1 < size[3])
			br[pri + size[0] + size[1] + size[2] + tid + 1] += err/16;
	}
	return;
}

void ditherimage(int height, int width, int intervalLen, float* in_cpu, float* in, unsigned char out_cpu[], unsigned char* out, int primals[])
{
	int size[4];
	bool right[3];
	int* g_size;
	bool* g_right;
	bool isGPU = false;
	float* zero_memory;
	cudaHostGetDevicePointer(&zero_memory, in_cpu,0);
	cudaMemcpy(in_cpu, in, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMalloc(&g_size, 4*sizeof(int));
	cudaMalloc(&g_right, 3*sizeof(bool));

	cudaStream_t mem;
	cudaStreamCreate(&mem);

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
		cudaMemcpy(g_right, right, 3*sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(g_size, size, 4*sizeof(int), cudaMemcpyHostToDevice);
		if(size[0] > 100)
		{
			if(!isGPU)
			{
				cudaMemcpy(in, in_cpu, width*height*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(out, out_cpu, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
				isGPU = true;
			}
			int n1 = (pb.col - 1 - cpu_width)/2 + 1;
			int no_threads = max(n1, size[0]);
			if(no_threads < 1024)
			{
				dither<<<1,no_threads>>>(g_right, primals[i-1], intervalLen, pb.col-1, cpu_width, in, out, g_size, zero_memory);
			}
			else
			{
				dither<<<ceil(((float)no_threads)/1024),1024>>>(g_right, primals[i-1], intervalLen, pb.col-1, cpu_width, in, out, g_size, zero_memory);
			}
			int j;
			for (j=size[0]-1; j>=0; j--)
			{
				if (pb.col - 1 - 2*j >= cpu_width)
				{
					break;
				}
				dither_cpu(right, primals[i-1], intervalLen, in_cpu, out_cpu, size, j);
			}
			cudaDeviceSynchronize();
			cudaMemcpyAsync(out_cpu + primals[i-1], out + primals[i-1], (j+1)*sizeof(unsigned char), cudaMemcpyDeviceToHost, mem);
		}
		else
		{
			if(isGPU)
			{
				cudaMemcpy(in_cpu, in, width*height*sizeof(float), cudaMemcpyDeviceToHost);
				isGPU = false;
			}
		
			for(int j=0; j<size[0]; j++)
			{
				dither_cpu(right, primals[i-1], intervalLen, in_cpu, out_cpu, size, j);
			}
		}
	}

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

	unsigned char* d_img = (unsigned char*)calloc(img_size, sizeof(unsigned char));
	unsigned char* pre[height];
	float *reordered;
	cudaHostAlloc(&reordered, width*height*sizeof(float), cudaHostAllocMapped);
	unsigned char* dithered = (unsigned char*)malloc(width*height*sizeof(unsigned char));
	float *g_reordered;
	unsigned char* g_dithered;
	int primals[width+2*(height-1)+4];
	for(int i=0; i<4; i++)
	{
		primals[width+2*(height-1) + i]=width*height;
	}
	cudaMalloc(&g_reordered, width*height*sizeof(float));
	cudaMalloc(&g_dithered, width*height*sizeof(unsigned char));
	if (d_img == NULL)
	{
		printf("Unable to allocate memory for image\n");
	}

	for(int i=0; i < height; i++)
	{
		pre[i] = img + i*width*channels;
	}

	reorder(height, width, channels, pre, reordered, primals);
	cudaMemcpy(g_reordered, reordered, width*height*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	ditherimage(height, width, intervalLen, reordered, g_reordered, dithered, g_dithered, primals);
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken in milliseconds: %f\n",milliseconds);

	order(height, width, channels, dithered, img, d_img);
	stbi_write_png(argv[3], width, height, channels, d_img, width*channels);
	free(d_img);
	cudaFree(g_dithered);
	cudaFree(g_reordered);

	return 0;
}
