#if !defined(__GNUC__) && !defined(__clang__)
#error "mastertools: gcc or clang required."
#endif

#ifndef MASTERTOOLS_VEC3_H_
#define MASTERTOOLS_VEC3_H_

#include <immintrin.h>

typedef union mvec3 mvec3;
union mvec3 {
    struct
    {
        float x, y, z, _pad1;
    };
    struct
    {
        float r, g, b, _pad2;
    };
    float _layout[4];
    __m128 vec;
} __attribute__((aligned(16)));

static inline mvec3 mt_vec3(float x, float y, float z)
{
    return (mvec3){.vec = _mm_setr_ps(x, y, z, 0.f)};
}

static inline mvec3 mt_vec3_zero(void)
{
    return (mvec3){.vec = _mm_setzero_ps()};
}

static inline mvec3 mt_vec3_all(float f)
{
    return (mvec3){.vec = _mm_blend_ps(_mm_set1_ps(f), _mm_setzero_ps(), 0x8)};
}

static inline mvec3 mt_vec3_add(mvec3 a, mvec3 b)
{
    return (mvec3){.vec = _mm_add_ps(a.vec, b.vec)};
}

static inline mvec3 mt_vec3_sub(mvec3 a, mvec3 b)
{
    return (mvec3){.vec = _mm_sub_ps(a.vec, b.vec)};
}

static inline mvec3 mt_vec3_mul(mvec3 a, mvec3 b)
{
    return (mvec3){.vec = _mm_mul_ps(a.vec, b.vec)};
}

static inline mvec3 mt_vec3_div(mvec3 a, mvec3 b)
{

    return (mvec3){.vec = _mm_div_ps(a.vec, b.vec)};
}

static inline mvec3 mt_vec3_negate(mvec3 a)
{
    return (mvec3){.vec = _mm_mul_ps(a.vec, _mm_set1_ps(-1.f))};
}

static inline float mt_vec3_dot(mvec3 a, mvec3 b)
{
    return _mm_cvtss_f32(_mm_dp_ps(a.vec, b.vec, 0x71));
}

static inline mvec3 mt_vec3_cross(mvec3 a, mvec3 b)
{
    __m128 shuff_a = _mm_shuffle_ps(a.vec, a.vec, 0xd2);
    __m128 shuff_b = _mm_shuffle_ps(b.vec, b.vec, 0xc9);
    __m128 product = _mm_mul_ps(shuff_a, shuff_b);

    shuff_a = _mm_shuffle_ps(a.vec, a.vec, 0xc9);
    shuff_b = _mm_shuffle_ps(b.vec, b.vec, 0xd2);

    return (mvec3){.vec = _mm_fmsub_ps(shuff_a, shuff_b, product)};
}

static inline float mt_vec3_length(mvec3 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0x71);
    return _mm_cvtss_f32(_mm_sqrt_ps(len2));
}

static inline mvec3 mt_vec3_normalize(mvec3 a)
{
    __m128 len2 = _mm_dp_ps(a.vec, a.vec, 0x7f);
    __m128 inv_len = _mm_rsqrt_ps(len2);
    return (mvec3){.vec = _mm_mul_ps(a.vec, inv_len)};
}

static inline mvec3 mt_vec3_inv(mvec3 a)
{
    return (mvec3){.vec = _mm_rcp_ps(a.vec)};
}

static inline mvec3 mt_vec3_fma(mvec3 a, mvec3 b, mvec3 c)
{
    return (mvec3){.vec = _mm_fmadd_ps(a.vec, b.vec, c.vec)};
}

static inline mvec3 mt_vec3_fms(mvec3 a, mvec3 b, mvec3 c)
{
    return (mvec3){.vec = _mm_fmsub_ps(a.vec, b.vec, c.vec)};
}

static inline mvec3 mt_vec3_load(const float *src)
{
    return (mvec3){.vec = _mm_blend_ps(_mm_loadu_ps(src), _mm_setzero_ps(), 0x8)};
}

#define mt_vec3_store(v, dest) _mm_maskstore_ps((float *)(dest), _mm_setr_epi32(-1, -1, -1, 0), (v).vec)

#endif