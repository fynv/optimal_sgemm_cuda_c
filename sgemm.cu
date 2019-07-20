#define USE_CUBLAS 0
#define CPU_REFERENCE 0

#include <stdlib.h>
#include <cuda_runtime.h>

#if USE_CUBLAS
#include <cublas_v2.h>
#endif

#include <vector>

#define VERBOSE 1
#include "Timing.h"

struct f8
{
	float4 a,b;
	__device__ inline f8()
	{
		memset(this, 0, sizeof(f8));
	}
};

struct f88
{
	f8 a,b,c,d,e,f,g,h;
};

__device__ inline void d_load8(const float* p, f8& c)
{
	c.a = ((float4*)p)[0];
	c.b = ((float4*)p)[16];
}

__device__ inline void d_store8(float* p, const f8& c)
{
	((float4*)p)[0] = c.a;
	((float4*)p)[16] = c.b;
}


__device__ inline void d_mult8v(f8& c, const f8& a, float b)
{
	c.a.x += a.a.x*b;
	c.a.y += a.a.y*b;
	c.a.z += a.a.z*b;
	c.a.w += a.a.w*b;
	c.b.x += a.b.x*b;
	c.b.y += a.b.y*b;
	c.b.z += a.b.z*b;
	c.b.w += a.b.w*b;
}

template<typename T>
__device__ inline void Swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

__global__ void 
__launch_bounds__(256)
g_fgemm (float* d_C, const float* d_A, const float* d_B, int n, int lda, int ldb, int ldc )
{
	int x_a = threadIdx.x & 31;
	int y_a = threadIdx.x >> 5;

	int x_b = threadIdx.x & 1;
	int y_b = threadIdx.x >> 1;

	int x_c = threadIdx.x & 15;
	int y_c = threadIdx.x >> 4;

	__shared__ float smem[4096];
	float *s_A1 = smem;
	float *s_A2 = smem + 1024;	
	float *s_B1 = smem + 2048;
	float *s_B2 = smem + 3072;

	f88 l_C;

	const float *p_A = d_A + (blockIdx.x<<7);
	const float *p_B = d_B + (blockIdx.y<<7) * ldb;


	float4 p, q;
	p = ((float4*)p_A)[y_a*(lda>>2) + x_a];
	q = ((float4*)p_B)[y_b*(ldb>>2) + x_b];

	for (int i=0; i<n; i+=8)
	{
		((float4*)s_A1)[threadIdx.x] = p;
		s_B1[(((x_b<<2) + 0)<<7) + y_b] = q.x;
		s_B1[(((x_b<<2) + 1)<<7) + y_b] = q.y;
		s_B1[(((x_b<<2) + 2)<<7) + y_b] = q.z;
		s_B1[(((x_b<<2) + 3)<<7) + y_b] = q.w;
		__syncthreads();    

		if (i+8<n)
		{
			p_A+=(lda<<3);
			p_B+=8;
			p = ((float4*)p_A)[y_a*(lda>>2) + x_a];
			q = ((float4*)p_B)[y_b*(ldb>>2) + x_b];
		}

		for (int j = 0; j<8; j++)
		{
			float *p_s_A = s_A1 + (j<<7) + (x_c<<2);
			float *p_s_B = s_B1 + (j<<7) + (y_c<<2);

			f8 a, b;
			d_load8(p_s_A, a);
			d_load8(p_s_B, b);

			d_mult8v(l_C.a, a, b.a.x);
			d_mult8v(l_C.b, a, b.a.y);
			d_mult8v(l_C.c, a, b.a.z);
			d_mult8v(l_C.d, a, b.a.w);
			d_mult8v(l_C.e, a, b.b.x);
			d_mult8v(l_C.f, a, b.b.y);
			d_mult8v(l_C.g, a, b.b.z);
			d_mult8v(l_C.h, a, b.b.w);
		}

		Swap(s_A1, s_A2);
		Swap(s_B1, s_B2);
	}

	float *p_C = d_C + ((blockIdx.x<<7)+(x_c<<2)) + ((blockIdx.y<<7)+(y_c<<2))*ldc;
	d_store8(p_C, l_C.a); p_C+=ldc;
	d_store8(p_C, l_C.b); p_C+=ldc;
	d_store8(p_C, l_C.c); p_C+=ldc;
	d_store8(p_C, l_C.d); p_C+=(ldc*61);
	d_store8(p_C, l_C.e); p_C+=ldc;
	d_store8(p_C, l_C.f); p_C+=ldc;
	d_store8(p_C, l_C.g); p_C+=ldc;
	d_store8(p_C, l_C.h); 

}


float rand01()
{
	return (float)rand()/(float)RAND_MAX;
}


int main()
{

	std::vector<float> h_A(1024*1024);
	std::vector<float> h_B(1024*1024);
	std::vector<float> h_C(1024*1024);

	for (unsigned i=0; i<1024*1024; i++)
	{
		h_A[i] = rand01();
		h_B[i] = rand01();
	}


#if USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif

	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc(&d_A, sizeof(float)*1024*1024);
	cudaMalloc(&d_B, sizeof(float)*1024*1024);
	cudaMalloc(&d_C, sizeof(float)*1024*1024);

	cudaMemcpy(d_A, h_A.data(), sizeof(float)*1024*1024, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), sizeof(float)*1024*1024, cudaMemcpyHostToDevice);


#if USE_CUBLAS
	{
		ScopedTimer _("cublasSgemm");

		float alpha = 1.0;
		float beta = 0.0;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024, &alpha, d_A, 1024, d_B, 1024, &beta, d_C, 1024);
	}
#else
	{
		ScopedTimer _("g_fgemm");
		g_fgemm <<< {1024/128, 1024/128, 1}, {256,1,1} >>> (d_C, d_A, d_B, 1024,1024,1024,1024);
	}	
#endif

	cudaMemcpy(h_C.data(), d_C, sizeof(float)*1024*1024, cudaMemcpyDeviceToHost);

	{
		FILE *fp= fopen("dump_gpu.txt", "w");
		for (unsigned i=0; i<1024*1024; i++)
			fprintf(fp, "%f\n", h_C[i]);
		fclose(fp);
	}

#if CPU_REFERENCE
	printf("running cpu reference.\n");
	for (unsigned i = 0; i< 1024; i++)
	{
		if (i%10==1)
			printf("col %d\n", i);
		for (unsigned j = 0; j<1024; j++)
		{
			float c=0.0;
			for (unsigned k = 0; k<1024; k++)
			{
				c+= h_A[j+k*1024] * h_B[i*1024+k];
			}
			h_C[i*1024+j]=c;
		}
	}

	{
		FILE *fp= fopen("dump_cpu.txt", "w");
		for (unsigned i=0; i<1024*1024; i++)
			fprintf(fp, "%f\n", h_C[i]);
		fclose(fp);
	}
#endif

	return 0;


}