/*
 * hmq algorithm built on cbuchner1's original X11
 *
 */

extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"

#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"

#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

extern uint32_t hmq_filter_cpu_sm2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch2);
extern void hmq_merge_cpu_sm2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_branch2);

static uint32_t *d_hash[MAX_GPUS];
static uint32_t* d_hash_br2[MAX_GPUS];  // SM 2

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
static uint32_t *d_branch1Nonces[MAX_GPUS];
static uint32_t *d_branch2Nonces[MAX_GPUS];
static uint32_t *d_branch3Nonces[MAX_GPUS];

extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void hmq_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void hmq_compactTest_cpu_free(int thr_id);
extern void hmq_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
	uint32_t *d_nonces1, uint32_t *nrm1, uint32_t *d_nonces2, uint32_t *nrm2, int order);
extern void hmq_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
	uint32_t *d_nonces1, uint32_t *nrm1, int order);

// hmq Hashfunktion
extern "C" void hmqhash(void *output, const void *input)
{
	// blake1-bmw2-grs3-skein4-jh5-keccak6-luffa7-cubehash8-shavite9-simd10-echo11-hamsi12-fugue13-shabal14-whirlpool15-sha512-haval17

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;
	sph_haval256_5_context ctx_haval;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, input, 80);
	sph_bmw512_close(&ctx_bmw, hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	if (hash[0] & 0x18)
	{
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512(&ctx_groestl, (const void*)hash, 64);
		sph_groestl512_close(&ctx_groestl, (void*)hash);
	}
	else
	{
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, (const void*)hash, 64);
		sph_skein512_close(&ctx_skein, (void*)hash);
	}

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, hash);

	if (hash[0] & 0x18)
	{
		sph_blake512_init(&ctx_blake);
		sph_blake512(&ctx_blake, (const void*)hash, 64);
		sph_blake512_close(&ctx_blake, (void*)hash);
	}
	else
	{
		sph_bmw512_init(&ctx_bmw);
		sph_bmw512(&ctx_bmw, (const void*) hash, 64);
		sph_bmw512_close(&ctx_bmw, hash);
	}

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, hash);

	if (hash[0] & 0x18)
	{
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512(&ctx_keccak, (const void*)hash, 64);
		sph_keccak512_close(&ctx_keccak, hash);
	}
	else
	{
		sph_jh512_init(&ctx_jh);
		sph_jh512(&ctx_jh, (const void*)hash, 64);
		sph_jh512_close(&ctx_jh, hash);
	}

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, hash);

	if (hash[0] & 0x18)
	{
		sph_whirlpool_init(&ctx_whirlpool);
		sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
		sph_whirlpool_close(&ctx_whirlpool, hash);
	}
	else
	{
		sph_haval256_5_init(&ctx_haval);
		sph_haval256_5(&ctx_haval, (const void*)hash, 64);
		memset(hash, 0, sizeof hash);
		sph_haval256_5_close(&ctx_haval, hash);
//		memset(&hash[8], 0, 32);
	}

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, hash);

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, (const void*)hash, 64);
	sph_blake512_close(&ctx_blake, (void*)hash);

	if (hash[0] & 0x18)
	{
		sph_shavite512_init(&ctx_shavite);
		sph_shavite512(&ctx_shavite, (const void*)hash, 64);
		sph_shavite512_close(&ctx_shavite, hash);
	}
	else
	{
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, (const void*)hash, 64);
		sph_luffa512_close(&ctx_luffa, hash);
	}

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	if (hash[0] & 0x18)
	{
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, (const void*)hash, 64);
		sph_echo512_close(&ctx_echo, hash);
	}
	else
	{
		sph_simd512_init(&ctx_simd);
		sph_simd512(&ctx_simd, (const void*)hash, 64);
		sph_simd512_close(&ctx_simd, hash);
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	if (hash[0] & 0x18)
	{
		sph_fugue512_init(&ctx_fugue);
		sph_fugue512(&ctx_fugue, (const void*)hash, 64);
		sph_fugue512_close(&ctx_fugue, hash);
	}
	else
	{
		sph_sha512_init(&ctx_sha512);
		sph_sha512(&ctx_sha512, (const void*)hash, 64);
		sph_sha512_close(&ctx_sha512, (void*)hash);
	}

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, (const void*)hash, 64);
	sph_sha512_close(&ctx_sha512, (void*)hash);


	if (hash[0] & 0x18)
	{
		sph_haval256_5_init(&ctx_haval);
		sph_haval256_5(&ctx_haval, (const void*)hash, 64);
		memset(hash, 0, sizeof hash);
		sph_haval256_5_close(&ctx_haval, hash);
//		memset(&hash[8], 0, 32);
	}
	else
	{
		sph_whirlpool_init(&ctx_whirlpool);
		sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
		sph_whirlpool_close(&ctx_whirlpool, hash);
	}

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, hash);

	memcpy(output, hash, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 32); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 32, cudaMemcpyDeviceToHost); \
		printf("quark %s %08x %08x %08x %08x...%08x... \n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3]), swab32(debugbuf[7])); \
		cudaFreeHost(debugbuf); \
		} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_hmq(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	uint32_t def_thr = 1U << 20; // 256*4096
	uint32_t throughput = cuda_default_throughput(thr_id, def_thr);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xffff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}

		cudaGetLastError();
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput));

		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x17_sha512_cpu_init(thr_id, throughput);
		x17_haval256_cpu_init(thr_id, throughput);
		hmq_compactTest_cpu_init(thr_id, throughput);


		if (cuda_arch[dev_id] >= 300) {
			cudaMalloc(&d_branch1Nonces[thr_id], sizeof(uint32_t)*throughput);
			cudaMalloc(&d_branch2Nonces[thr_id], sizeof(uint32_t)*throughput);
			cudaMalloc(&d_branch3Nonces[thr_id], sizeof(uint32_t)*throughput);
		}
		else {
			cudaMalloc(&d_hash_br2[thr_id], (size_t)64 * throughput);
		}

//		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);
		cuda_check_cpu_init(thr_id, throughput);
		CUDA_SAFE_CALL(cudaGetLastError());

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_bmw512_cpu_setBlock_80(endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		uint32_t foundNonce;
		uint32_t nrm1 = 0, nrm2 = 0, nrm3 = 0;

		// Hash with CUDA
		quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);;
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		if (cuda_arch[dev_id] >= 300) {

			hmq_compactTest_single_false_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
				d_branch3Nonces[thr_id], &nrm3, order++);

			quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			quark_jh512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			quark_keccak512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_bmw512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			x11_luffa512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			x11_cubehash512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			quark_keccak512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			x11_shavite512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			x11_simd512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			x15_whirlpool_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			x17_haval256_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			x11_echo512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			quark_blake512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			x11_shavite512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			x11_luffa512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			x13_hamsi512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			x13_fugue512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			x11_echo512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			x11_simd512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			x14_shabal512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			x15_whirlpool_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			x13_fugue512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			x17_sha512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			quark_groestl512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			x17_sha512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			hmq_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			x17_haval256_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			x15_whirlpool_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			quark_bmw512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			foundNonce = cuda_check_hash_branch(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
		} else {

			/* algo permutations are made with 2 different buffers */

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm1  :");

			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("jh512     :");
			quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("keccak512 :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm2    :");

			//x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			//TRACE("luffa512:");
			//x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
			TRACE("cube512   :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm3  :");

			x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("shavite512:");
			x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("simd512   :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm4  :");

			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("echo512   :");
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("blake512  :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm5  :");

			x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("hamsi512  :");
			x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("fugue512  :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm6  :");

			x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("shabal5512:");
			x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("whirlpool :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm7  :");

			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("groestl512:");
			x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("sha512    :");

			hmq_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			hmq_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm8  :");

			quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("bmw512    :");

			CUDA_LOG_ERROR();
			foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}

		*hashes_done = pdata[19] - first_nonce + (throughput / 4);

		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash[8];
			be32enc(&endiandata[19], foundNonce);
			hmqhash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				pdata[19] = foundNonce;
				return 1;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
				applog_hash((uchar*)vhash);
				applog_hash((uchar*)ptarget);
			}
		}

		//if ((uint64_t)throughput + pdata[19] >= max_nonce) {
		//	pdata[19] = max_nonce;
		//	break;
		//}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_hmq(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	cudaFree(d_branch1Nonces[thr_id]);
	cudaFree(d_branch2Nonces[thr_id]);
	cudaFree(d_branch3Nonces[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);
	hmq_compactTest_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
