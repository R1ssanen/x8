#if !defined(__GNUC__) && !defined(__clang__)
#error "mastertools: gcc or clang required."
#endif

#ifndef MASTERTOOLS_VEC4_H_
#define MASTERTOOLS_VEC4_H_

#include <immintrin.h>

typedef union mvec4 mvec4;
union mvec4 {
    struct
    {
        float x, y, z, w;
    };
    struct
    {
        float r, g, b, a;
    };
    float _layout[4];
    __m128 vec;
} __attribute__((aligned(16)));

static inline mvec4 mt_vec4(float x, float y, float z, float w)
{
    return (mvec4){.vec = _mm_setr_ps(x, y, z, w)};
}

static inline mvec4 mt_vec4_zero(void)
{
    return (mvec4){.vec = _mm_setzero_ps()};
}

static inline mvec4 mt_vec4_all(float f)
{
    return (mvec4){.vec = _mm_set1_ps(f)};
}

static inline mvec4 mt_vec4_add(mvec4 a, mvec4 b)
{
    return (mvec4){.vec = _mm_add_ps(a.vec, b.vec)};
}

static inline mvec4 mt_vec4_sub(mvec4 a, mvec4 b)
{
    return (mvec4){.vec = _mm_sub_ps(a.vec, b.vec)};
}

static inline mvec4 mt_vec4_mul(mvec4 a, mvec4 b)
{
    return (mvec4){.vec = _mm_mul_ps(a.vec, b.vec)};
}

static inline mvec4 mt_vec4_div(mvec4 a, mvec4 b)
{

    return (mvec4){.vec = _mm_div_ps(a.vec, b.vec)};
}

static inline mvec4 mt_vec4_negate(mvec4 a)
{
    return (mvec4){.vec = _mm_sub_ps(_mm_setzero_ps(), a.vec)};
}

static inline float mt_vec4_dot(mvec4 a, mvec4 b)
{
    return _mm_cvtss_f32(_mm_dp_ps(a.vec, b.vec, 0xf1));
}

static inline float mt_vec4_length(mvec4 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0xf1);
    return _mm_cvtss_f32(_mm_sqrt_ps(len2));
}

static inline mvec4 mt_vec4_normalize(mvec4 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0xff);
    __m128 inv_len = _mm_rsqrt_ps(len2);
    return (mvec4){.vec = _mm_mul_ps(a.vec, inv_len)};
}

static inline mvec4 mt_vec4_inv(mvec4 a)
{
    return (mvec4){.vec = _mm_rcp_ps(a.vec)};
}

static inline mvec4 mt_vec4_fma(mvec4 a, mvec4 b, mvec4 c)
{
    return (mvec4){.vec = _mm_fmadd_ps(a.vec, b.vec, c.vec)};
}

static inline mvec4 mt_vec4_load(const float *src)
{
    return (mvec4){.vec = _mm_loadu_ps(src)};
}

static inline void mt_vec4_store(mvec4 a, float *dest)
{
    _mm_storeu_ps(dest, a.vec);
}

#endif