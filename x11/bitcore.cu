/**
 * Timetravel-10 (bitcore) CUDA implementation
 *  by tpruvot@github - May 2017
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

#define HASH_FUNC_BASE_TIMESTAMP 1492973331U
#define HASH_FUNC_COUNT 10
#define HASH_FUNC_COUNT_PERMUTATIONS 40320U

extern "C" {
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
#if HASH_FUNC_COUNT > 10
#include "sph/sph_echo.h"
#endif
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t permutations[HASH_FUNC_COUNT_PERMUTATIONS] = {
	#include "timetravel-permutations.h"
};

enum Algo {
//	BLAKE = 0,
//	BMW,
	GROESTL = 0,
	SKEIN,
	JH,
	KECCAK,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	MAX_ALGOS_COUNT
};

#define INITIAL_DATE HASH_FUNC_BASE_TIMESTAMP
static inline uint32_t getCurrentAlgoSeq(uint32_t ntime)
{
	// unlike x11evo, the permutation changes often (with ntime)
	return (uint32_t) (ntime - INITIAL_DATE) % HASH_FUNC_COUNT_PERMUTATIONS;
}

// CPU Hash
extern "C" void bitcore_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[64/4] = { 0 };

	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_skein512_context     ctx_skein;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_luffa512_context     ctx_luffa1;
	sph_cubehash512_context  ctx_cubehash1;
	sph_shavite512_context   ctx_shavite1;
	sph_simd512_context      ctx_simd1;

	uint32_t *data = (uint32_t*)input;
	const uint32_t ntime = (opt_benchmark || !data[17]) ? (uint32_t)time(NULL) : data[17];
	const uint32_t sequence = permutations[getCurrentAlgoSeq(ntime)];

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hash, 64);
	sph_bmw512_close(&ctx_bmw, hash);

	for (int i = 0; i < (4 * (HASH_FUNC_COUNT - 2)); i += 4)
	{
		switch ((sequence >> i) & 0xf) {
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, hash, 64);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, hash, 64);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, hash, 64);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, hash, 64);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa1);
			sph_luffa512(&ctx_luffa1, hash, 64);
			sph_luffa512_close(&ctx_luffa1, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash1);
			sph_cubehash512(&ctx_cubehash1, hash, 64);
			sph_cubehash512_close(&ctx_cubehash1, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite1);
			sph_shavite512(&ctx_shavite1, hash, 64);
			sph_shavite512_close(&ctx_shavite1, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd1);
			sph_simd512(&ctx_simd1, hash, 64);
			sph_simd512_close(&ctx_simd1, hash);
			break;
		}
	}

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "tt-"
#include "cuda_debug.cuh"

void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_bitcore(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark) pdata[17] = swab32(0x59090909);

	uint32_t ntime = swab32(work->data[17]);
	uint32_t sequence = permutations[getCurrentAlgoSeq(ntime)];

	if (opt_benchmark)
		ptarget[7] = 0x5;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0) {
			return 0;
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMemset(d_hash[thr_id], 0, (size_t) 64 * throughput), -1);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	// first and second algo seems locked to blake+bmw in bitcore, fine!
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("blake  :");
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("bmw    :");

		for (int i = 0; i < (4 * (HASH_FUNC_COUNT - 2)); i += 4)
		{
			switch ((sequence >> i) & 0xf) {
			case GROESTL:
				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("groestl:");
				break;
			case SKEIN:
				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("skein  :");
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("jh512  :");
				break;
			case KECCAK:
				quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("keccak :");
				break;
			case LUFFA:
				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("luffa  :");
				break;
			case CUBEHASH:
				x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("cube   :");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("shavite:");
				break;
			case SIMD:
				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("simd   :");
				break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			be32enc(&endiandata[19], work->nonces[0]);
			bitcore_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				pdata[19] = work->nonces[0];
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					bitcore_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					}
					pdata[19] = max(pdata[19], work->nonces[1]) + 1;
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_bitcore(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
