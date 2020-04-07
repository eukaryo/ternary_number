/*
MIT License
Copyright (c) 2020 Hiroki Takizawa
*/
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>


//2進40桁の数xを、3進数と解釈した値を返す。
//ちなみに2進40桁の数を3進数として解釈すると必ず2^64未満になる。例えばpythonで書くと
//sum([3**x for x in range(40)]) < 2**64
uint64_t ternarize(uint64_t x) {

	assert(x <= 0x0000'00FF'FFFF'FFFFULL);

	uint64_t answer = 0;

	for (uint64_t value = 1; x; x /= 2, value *= 3) {
		if (x & 1ULL) {
			answer += value;
		}
	}

	return answer;
}

//3進40桁の数をビットプレーン分解してuとlにしたとする。この3進数を2進数に変換した値を返す。
//ちなみに3進40桁の数は必ず2^64未満になる。例えばpythonで書くと
//sum([2*(3**x) for x in range(40)]) < 2**64
uint64_t ternarize_naive1(uint64_t u, uint64_t l) {

	assert((u | l) <= 0x0000'00FF'FFFF'FFFFULL);
	assert((u & l) == 0);

	return ternarize(u) * 2 + ternarize(l);
}
uint64_t ternarize_naive2(uint64_t u, uint64_t l) {

	assert((u | l) <= 0x0000'00FF'FFFF'FFFFULL);
	assert((u & l) == 0);

	uint64_t answer = 0;

	for (uint64_t value = 1; u | l; u /= 2, l /= 2, value *= 3) {
		if (u & 1ULL) {
			answer += value * 2;
		}
		else if (l & 1ULL) {
			answer += value;
		}
	}

	return answer;
}
uint64_t ternarize_sse(uint64_t u, uint64_t l) {

	assert((u | l) <= 0x0000'00FF'FFFF'FFFFULL);
	assert((u & l) == 0);

	const __m128i ternary_binary_table = _mm_set_epi8(40, 39, 37, 36, 31, 30, 28, 27, 13, 12, 10, 9, 4, 3, 1, 0);
	const __m128i bitmask0F = _mm_set1_epi8(0x0F);

	const __m128i lu_lo = _mm_set_epi64x(l, u);
	const __m128i lu_hi = _mm_srli_epi64(lu_lo, 4);

	const __m128i u_4bits = _mm_and_si128(_mm_unpacklo_epi8(lu_lo, lu_hi), bitmask0F);
	const __m128i l_4bits = _mm_and_si128(_mm_unpackhi_epi8(lu_lo, lu_hi), bitmask0F);

	//この時点でu_4bitsは、uを4bit区切りにして__m128iの8bit領域16箇所に順番に詰めた形になっている。l_4bitsも同様。

	const __m128i ternarized_u_4bits = _mm_shuffle_epi8(ternary_binary_table, u_4bits);
	const __m128i ternarized_l_4bits = _mm_shuffle_epi8(ternary_binary_table, l_4bits);
	const __m128i answer_base_3p4_in_i8s = _mm_add_epi8(_mm_add_epi8(ternarized_u_4bits, ternarized_u_4bits), ternarized_l_4bits);

	//この時点でanswer_base_3p4_in_i8sは、真の値を"81進数"にした状態で、各桁の値を__m128iの8bit領域16箇所に順番に詰めた形になっている。

	const __m128i tmp_mask8_lo = _mm_set_epi8(0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF);
	const __m128i tmp_shuf8_hi = _mm_set_epi8(0xFF, 15, 0xFF, 13, 0xFF, 11, 0xFF, 9, 0xFF, 7, 0xFF, 5, 0xFF, 3, 0xFF, 1);
	const __m128i i16_81s = _mm_set1_epi16(81);
	const __m128i answer_tmp8_lo = _mm_and_si128(answer_base_3p4_in_i8s, tmp_mask8_lo);
	const __m128i answer_tmp8_hi = _mm_shuffle_epi8(answer_base_3p4_in_i8s, tmp_shuf8_hi);
	const __m128i answer_base3p8_in_i16s = _mm_add_epi16(answer_tmp8_lo, _mm_mullo_epi16(answer_tmp8_hi, i16_81s));

	//この時点でanswer_base3p8_in_i16sは、真の値を"6561進数"(6561=81*81=3^8)にした状態で、各桁の値を__m128iの16bit領域8箇所に順番に詰めた形になっている。

	const __m128i tmp_mask16_lo = _mm_set_epi8(0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF);
	const __m128i tmp_shuf16_hi = _mm_set_epi8(0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 3, 2);
	const __m128i i32_6561s = _mm_set1_epi32(6561);
	const __m128i answer_tmp16_lo = _mm_and_si128(answer_base3p8_in_i16s, tmp_mask16_lo);
	const __m128i answer_tmp16_hi = _mm_shuffle_epi8(answer_base3p8_in_i16s, tmp_shuf16_hi);
	const __m128i answer_base3p16_in_i32s = _mm_add_epi32(answer_tmp16_lo, _mm_mullo_epi32(answer_tmp16_hi, i32_6561s));

	//この時点でanswer_base3p16_in_i32sは、真の値を"43046721進数"(43046721=6561*6561=3^16)にした状態で、各桁の値を__m128iの32bit領域4箇所に順番に詰めた形になっている。

	const __m128i tmp_mask32_lo = _mm_set_epi8(0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m128i tmp_shuf32_hi = _mm_set_epi8(0xFF, 0xFF, 0xFF, 0xFF, 15, 14, 13, 12, 0xFF, 0xFF, 0xFF, 0xFF, 7, 6, 5, 4);
	const __m128i i64_43046721s = _mm_set1_epi64x(43046721);
	const __m128i answer_tmp32_lo = _mm_and_si128(answer_base3p16_in_i32s, tmp_mask32_lo);
	const __m128i answer_tmp32_hi = _mm_shuffle_epi8(answer_base3p16_in_i32s, tmp_shuf32_hi);
	const __m128i answer_base3p32_in_i64s = _mm_add_epi64(answer_tmp32_lo, _mm_mul_epi32(answer_tmp32_hi, i64_43046721s));

	//この時点でanswer_base3p32_in_i64sは、真の値を"1853020188851841進数"にした状態で、各桁の値を__m128iの64bit領域2箇所に順番に詰めた形になっている。
	//ちなみに1853020188851841<2^63なのでepi64で扱っても問題ない。

	alignas(16) uint64_t answer[2] = {};
	_mm_storeu_si128((__m128i*)answer, answer_base3p32_in_i64s);
	return answer[1] * 1853020188851841ULL + answer[0];
}

void ternarize_naive1_full(const uint64_t u, const uint64_t l, uint64_t *answer_l, uint64_t *answer_u) {

	assert((u & l) == 0);

	*answer_l = ternarize(u & 0x0000'00FF'FFFF'FFFFULL) * 2 + ternarize(l & 0x0000'00FF'FFFF'FFFFULL);
	*answer_u = ternarize(u >> 40ULL) * 2 + ternarize(l >> 40ULL);
}
void ternarize_naive2_full(const uint64_t u, const uint64_t l, uint64_t *answer_l, uint64_t *answer_u) {

	assert((u & l) == 0);

	*answer_l = 0;
	*answer_u = 0;

	for (uint64_t value = 1, ul = u & 0x0000'00FF'FFFF'FFFFULL, ll = l & 0x0000'00FF'FFFF'FFFFULL; ul | ll; ul /= 2, ll /= 2, value *= 3) {
		if (ul & 1ULL) {
			*answer_l += value * 2;
		}
		else if (ll & 1ULL) {
			*answer_l += value;
		}
	}
	for (uint64_t value = 1, uu = u >> 40ULL, lu = l >> 40ULL; uu | lu; uu /= 2, lu /= 2, value *= 3) {
		if (uu & 1ULL) {
			*answer_u += value * 2;
		}
		else if (lu & 1ULL) {
			*answer_u += value;
		}
	}
}
void ternarize_avx2_full(const uint64_t u, const uint64_t l, uint64_t *answer_l, uint64_t *answer_u) {

	assert((u & l) == 0);

	const __m256i ternary_binary_table = _mm256_set_epi8(40, 39, 37, 36, 31, 30, 28, 27, 13, 12, 10, 9, 4, 3, 1, 0, 40, 39, 37, 36, 31, 30, 28, 27, 13, 12, 10, 9, 4, 3, 1, 0);
	const __m256i bitmask0F = _mm256_set1_epi8(0x0F);

	const __m256i lu_lo = _mm256_set_epi64x(l >> 40ULL, u >> 40ULL, l & 0x0000'00FF'FFFF'FFFFULL, u & 0x0000'00FF'FFFF'FFFFULL);
	const __m256i lu_hi = _mm256_srli_epi64(lu_lo, 4);

	const __m256i u_4bits = _mm256_and_si256(_mm256_unpacklo_epi8(lu_lo, lu_hi), bitmask0F);
	const __m256i l_4bits = _mm256_and_si256(_mm256_unpackhi_epi8(lu_lo, lu_hi), bitmask0F);

	const __m256i ternarized_u_4bits = _mm256_shuffle_epi8(ternary_binary_table, u_4bits);
	const __m256i ternarized_l_4bits = _mm256_shuffle_epi8(ternary_binary_table, l_4bits);
	const __m256i answer_base_3p4_in_i8s = _mm256_add_epi8(_mm256_add_epi8(ternarized_u_4bits, ternarized_u_4bits), ternarized_l_4bits);

	const __m256i tmp_mask8_lo = _mm256_set_epi8(0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF);
	const __m256i tmp_shuf8_hi = _mm256_set_epi8(0xFF, 15, 0xFF, 13, 0xFF, 11, 0xFF, 9, 0xFF, 7, 0xFF, 5, 0xFF, 3, 0xFF, 1, 0xFF, 15, 0xFF, 13, 0xFF, 11, 0xFF, 9, 0xFF, 7, 0xFF, 5, 0xFF, 3, 0xFF, 1);
	const __m256i i16_81s = _mm256_set1_epi16(81);
	const __m256i answer_tmp8_lo = _mm256_and_si256(answer_base_3p4_in_i8s, tmp_mask8_lo);
	const __m256i answer_tmp8_hi = _mm256_shuffle_epi8(answer_base_3p4_in_i8s, tmp_shuf8_hi);
	const __m256i answer_base3p8_in_i16s = _mm256_add_epi16(answer_tmp8_lo, _mm256_mullo_epi16(answer_tmp8_hi, i16_81s));

	const __m256i tmp_mask16_lo = _mm256_set_epi8(0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF);
	const __m256i tmp_shuf16_hi = _mm256_set_epi8(0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 3, 2, 0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 3, 2);
	const __m256i i32_6561s = _mm256_set1_epi32(6561);
	const __m256i answer_tmp16_lo = _mm256_and_si256(answer_base3p8_in_i16s, tmp_mask16_lo);
	const __m256i answer_tmp16_hi = _mm256_shuffle_epi8(answer_base3p8_in_i16s, tmp_shuf16_hi);
	const __m256i answer_base3p16_in_i32s = _mm256_add_epi32(answer_tmp16_lo, _mm256_mullo_epi32(answer_tmp16_hi, i32_6561s));

	const __m256i tmp_mask32_lo = _mm256_set_epi8(0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m256i tmp_shuf32_hi = _mm256_set_epi8(0xFF, 0xFF, 0xFF, 0xFF, 15, 14, 13, 12, 0xFF, 0xFF, 0xFF, 0xFF, 7, 6, 5, 4, 0xFF, 0xFF, 0xFF, 0xFF, 15, 14, 13, 12, 0xFF, 0xFF, 0xFF, 0xFF, 7, 6, 5, 4);
	const __m256i i64_43046721s = _mm256_set1_epi64x(43046721);
	const __m256i answer_tmp32_lo = _mm256_and_si256(answer_base3p16_in_i32s, tmp_mask32_lo);
	const __m256i answer_tmp32_hi = _mm256_shuffle_epi8(answer_base3p16_in_i32s, tmp_shuf32_hi);
	const __m256i answer_base3p32_in_i64s = _mm256_add_epi64(answer_tmp32_lo, _mm256_mul_epi32(answer_tmp32_hi, i64_43046721s));

	alignas(32) uint64_t answer[4] = {};
	_mm256_storeu_si256((__m256i*)answer, answer_base3p32_in_i64s);
	*answer_l = answer[1] * 1853020188851841ULL + answer[0];
	*answer_u = answer[2];
	assert(answer[3] == 0);
}

static uint64_t xorshift64_state = 0x3141592653589793ULL;
uint64_t xorshift64() {
	xorshift64_state = xorshift64_state ^ (xorshift64_state << 7);
	return xorshift64_state ^ (xorshift64_state >> 9);
}

constexpr int NUM_REPEAT = 1 << 30;

void test_40bit() {

	std::cout << "start: test_40bit" << std::endl;
	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "negative control:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			volatile uint64_t l = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			l ^= (u & l);
			result += l;
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << /*"result (for validation) = " << result << std::endl <<*/ std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "naive version1:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			volatile uint64_t l = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			l ^= (u & l);
			result += ternarize_naive1(u, l);
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "naive version2:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			volatile uint64_t l = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			l ^= (u & l);
			result += ternarize_naive2(u, l);
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "SSE version:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			volatile uint64_t l = xorshift64() & 0x0000'00FF'FFFF'FFFFULL;
			l ^= (u & l);
			result += ternarize_sse(u, l);
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}
}

void test_64bit() {

	std::cout << "start: test_64bit" << std::endl;
	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "negative control:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64();
			volatile uint64_t l = xorshift64();
			l ^= (u & l);
			volatile uint64_t r0 = 0, r1 = 0;
			result += r0 + r1;
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << /*"result (for validation) = " << result << std::endl <<*/ std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "naive version1:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64();
			volatile uint64_t l = xorshift64();
			l ^= (u & l);
			uint64_t r0 = 0, r1 = 0;
			ternarize_naive1_full(u, l, &r0, &r1);
			result += r0 + r1;
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "naive version2:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64();
			volatile uint64_t l = xorshift64();
			l ^= (u & l);
			uint64_t r0 = 0, r1 = 0;
			ternarize_naive2_full(u, l, &r0, &r1);
			result += r0 + r1;
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}

	{
		xorshift64_state = 0x3141592653589793ULL;
		std::cout << "AVX2 version:" << std::endl;
		volatile uint64_t result = 0;
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < NUM_REPEAT; ++i) {
			volatile uint64_t u = xorshift64();
			volatile uint64_t l = xorshift64();
			l ^= (u & l);
			uint64_t r0 = 0, r1 = 0;
			ternarize_avx2_full(u, l, &r0, &r1);
			result += r0 + r1;
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;
		std::cout << "result (for validation) = " << result << std::endl << std::endl;
	}
}

int main() {

	test_40bit();
	test_64bit();

	return 0;
}