#if !defined(__GNUC__) && !defined(__clang__)
#error "mastertools: gcc or clang required."
#endif

#ifndef MASTERTOOLS_VEC2_H_
#define MASTERTOOLS_VEC2_H_

#include <immintrin.h>

typedef union mvec2 mvec2;
union mvec2 {
    struct
    {
        float x, y, _pad1, _pad2;
    };
    struct
    {
        float u, v, _pad3, _pad4;
    };
    float _layout[4];
    __m128 vec;
} __attribute__((aligned(16)));

static inline mvec2 mt_vec2(float x, float y)
{
    return (mvec2){.vec = _mm_setr_ps(x, y, 0.f, 0.f)};
}

static inline mvec2 mt_vec2_zero(void)
{
    return (mvec2){.vec = _mm_setzero_ps()};
}

static inline mvec2 mt_vec2_all(float f)
{
    const __m128 a = _mm_set1_ps(f);
    return (mvec2){.vec = _mm_movelh_ps(a, _mm_setzero_ps())};
}

static inline mvec2 mt_vec2_add(mvec2 a, mvec2 b)
{
    return (mvec2){.vec = _mm_add_ps(a.vec, b.vec)};
}

static inline mvec2 mt_vec2_sub(mvec2 a, mvec2 b)
{
    return (mvec2){.vec = _mm_sub_ps(a.vec, b.vec)};
}

static inline mvec2 mt_vec2_mul(mvec2 a, mvec2 b)
{
    return (mvec2){.vec = _mm_mul_ps(a.vec, b.vec)};
}

static inline mvec2 mt_vec2_div(mvec2 a, mvec2 b)
{

    return (mvec2){.vec = _mm_div_ps(a.vec, b.vec)};
}

static inline mvec2 mt_vec2_negate(mvec2 a)
{
    return (mvec2){.vec = _mm_mul_ps(a.vec, _mm_set1_ps(-1.f))};
}

static inline float mt_vec2_dot(mvec2 a, mvec2 b)
{
    return _mm_cvtss_f32(_mm_dp_ps(a.vec, b.vec, 0x31));
}

static inline float mt_vec2_length(mvec2 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0x31);
    return _mm_cvtss_f32(_mm_sqrt_ps(len2));
}

static inline mvec2 mt_vec2_normalize(mvec2 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0x3f);
    __m128 inv_len = _mm_rsqrt_ps(len2);
    return (mvec2){.vec = _mm_mul_ps(a.vec, inv_len)};
}

static inline mvec2 mt_vec2_inv(mvec2 a)
{
    return (mvec2){.vec = _mm_rcp_ps(a.vec)};
}

static inline mvec2 mt_vec2_fma(mvec2 a, mvec2 b, mvec2 c)
{
    return (mvec2){.vec = _mm_fmadd_ps(a.vec, b.vec, c.vec)};
}

static inline mvec2 mt_vec2_load(const float *src)
{
    return (mvec2){.vec = _mm_blend_ps(_mm_loadu_ps(src), _mm_setzero_ps(), 0xc)};
}

static inline void mt_vec2_store(mvec2 a, float *dest)
{
    _mm_storel_pi((__m64 *)dest, a.vec);
}

#endif