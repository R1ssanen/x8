#define ROHAN_DEFINE_SHADER
#define ROHAN_ATTRIB_COUNT 3
#define ROHAN_NO_DEPTH
#define ROHAN_USE_DISCARD
#include "rohan.h"

#include "math/vec3.h"
#include "texture.h"

typedef struct uniforms uniforms;
struct uniforms
{
    __m256 dx;
    __m256 dy;
    __m256 dz;

    mt_texture texture;
    mt_texture pixels;
    float *depth;
    uint32_t id;
    mt_texture *cubemap;
};

static inline __m256i sample_2d(__m256 u, __m256 v, const mt_texture *texture)
{
    __m256 fu = _mm256_mul_ps(u, _mm256_set1_ps(texture->width));
    __m256 fv = _mm256_mul_ps(v, _mm256_set1_ps(texture->height));

    __m256i iu = _mm256_and_si256(_mm256_cvttps_epi32(fu), _mm256_set1_epi32(texture->width - 1));
    __m256i iv = _mm256_and_si256(_mm256_cvttps_epi32(fv), _mm256_set1_epi32(texture->height - 1));

    //__m256i index = _mm256_add_epi32(iu, _mm256_slli_epi32(iv, texture->log2_width));
    __m256i index = _mm256_add_epi32(iu, _mm256_mullo_epi32(iv, _mm256_set1_epi32(texture->width)));
    return _mm256_i32gather_epi32((int *)texture->data, index, 4);
}

static inline __m256 abs_m256(__m256 x)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}

static inline __m256 neg_m256(__m256 x)
{
    return _mm256_sub_ps(_mm256_setzero_ps(), x);
}

static inline __m256 not_m256(__m256 x)
{
    return _mm256_xor_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
}

static inline __m256i sample_cubemap(__m256 dx, __m256 dy, __m256 dz, __m256 *u, __m256 *v)
{
    __m256 abs_x = abs_m256(dx), abs_y = abs_m256(dy), abs_z = abs_m256(dz);

    __m256 x_greater_y = _mm256_cmp_ps(abs_x, abs_y, _CMP_GE_OQ);
    __m256 max_x = _mm256_and_ps(x_greater_y, _mm256_cmp_ps(abs_x, abs_z, _CMP_GE_OQ));
    __m256 max_y = _mm256_and_ps(not_m256(x_greater_y), _mm256_cmp_ps(abs_y, abs_z, _CMP_GE_OQ));

    __m256 max_z = not_m256(_mm256_or_ps(max_x, max_y));

    const __m256 f_zero = _mm256_setzero_ps();
    __m256 pos_x = _mm256_cmp_ps(dx, f_zero, _CMP_GT_OQ);
    __m256 pos_y = _mm256_cmp_ps(dy, f_zero, _CMP_GT_OQ);
    __m256 pos_z = _mm256_cmp_ps(dz, f_zero, _CMP_GT_OQ);

    const __m256 f_sign = _mm256_set1_ps(-0.f);

    __m256 vc = _mm256_blendv_ps(neg_m256(dy), _mm256_xor_ps(dz, _mm256_and_ps(pos_y, f_sign)), max_y);

    __m256 uc = _mm256_blendv_ps(dx, neg_m256(dx), _mm256_and_ps(pos_z, max_z));
    uc = _mm256_blendv_ps(uc, _mm256_blendv_ps(neg_m256(dz), dz, pos_x), max_x);

    const __m256 f_half = _mm256_set1_ps(0.5f);
    __m256 denom = _mm256_max_ps(abs_x, _mm256_max_ps(abs_y, abs_z));
    __m256 half_inv_denom = _mm256_mul_ps(_mm256_rcp_ps(denom), f_half);

    *u = _mm256_fmadd_ps(half_inv_denom, uc, f_half);
    *v = _mm256_fmadd_ps(half_inv_denom, vc, f_half);

    __m256i addy = _mm256_srli_epi32(_mm256_castps_si256(pos_y), 31);
    __m256i addz = _mm256_srli_epi32(_mm256_castps_si256(pos_z), 31);

    __m256i fx = _mm256_srli_epi32(_mm256_castps_si256(pos_x), 31);
    __m256i fy = _mm256_add_epi32(_mm256_set1_epi32(2), addy);
    __m256i fz = _mm256_add_epi32(_mm256_set1_epi32(4), addz);

    return _mm256_blendv_epi8(_mm256_blendv_epi8(fz, fy, _mm256_castps_si256(max_y)), fx, _mm256_castps_si256(max_x));
}

static inline __m256i spread_bits(__m256i x)
{
    x = _mm256_and_si256(_mm256_or_si256(x, _mm256_slli_epi32(x, 8)), _mm256_set1_epi32(0x00FF00FF));
    x = _mm256_and_si256(_mm256_or_si256(x, _mm256_slli_epi32(x, 4)), _mm256_set1_epi32(0x0F0F0F0F));
    x = _mm256_and_si256(_mm256_or_si256(x, _mm256_slli_epi32(x, 2)), _mm256_set1_epi32(0x33333333));
    x = _mm256_and_si256(_mm256_or_si256(x, _mm256_slli_epi32(x, 1)), _mm256_set1_epi32(0x55555555));

    return x;
}

static inline __m256i reverse_bits32(__m256i x)
{
    x = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x55555555)), 1),
                        _mm256_and_si256(_mm256_srli_epi32(x, 1), _mm256_set1_epi32(0x55555555)));

    x = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x33333333)), 2),
                        _mm256_and_si256(_mm256_srli_epi32(x, 2), _mm256_set1_epi32(0x33333333)));

    x = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x0F0F0F0F)), 4),
                        _mm256_and_si256(_mm256_srli_epi32(x, 4), _mm256_set1_epi32(0x0F0F0F0F)));

    x = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x00FF00FF)), 8),
                        _mm256_and_si256(_mm256_srli_epi32(x, 8), _mm256_set1_epi32(0x00FF00FF)));

    return _mm256_or_si256(_mm256_slli_epi32(x, 16), _mm256_srli_epi32(x, 16));
}

static inline __m256 bayer_dither(__m256i i, __m256i j)
{
    const int K = 2;
    const int N = 1 << K;

    // Keep only the coordinate bits belonging to the matrix.
    const __m256i mask = _mm256_set1_epi32(N - 1);

    i = _mm256_and_si256(i, mask);
    j = _mm256_and_si256(j, mask);

    // Interleave (i ^ j) and i.
    __m256i a = spread_bits(_mm256_xor_si256(i, j));
    __m256i b = _mm256_slli_epi32(spread_bits(i), 1);

    __m256i morton = _mm256_or_si256(a, b);

    // Reverse only the meaningful 2*K bits.
    __m256i reversed = reverse_bits32(morton);
    reversed = _mm256_srli_epi32(reversed, 32 - 2 * K);

    // Normalize to [0, 1).
    __m256 value = _mm256_cvtepi32_ps(reversed);
    return _mm256_mul_ps(value, _mm256_set1_ps(1.0f / (N * N)));
}

__m256i frag_main(void *restrict instance, const rohan_raster_state *restrict state, __m256 *restrict allow)
{
    uniforms *uniform = instance;

    __m256 u, v;
    __m256i faces = sample_cubemap(uniform->dx, uniform->dy, uniform->dz, &u, &v);
    __m256i offset = _mm256_mullo_epi32(faces, _mm256_set1_epi32(uniform->cubemap->width * uniform->cubemap->height));
    __m256 fu = _mm256_fmadd_ps(u, _mm256_set1_ps(uniform->cubemap->width - 1.f), _mm256_set1_ps(0.5f));
    __m256 fv = _mm256_fmadd_ps(v, _mm256_set1_ps(uniform->cubemap->height - 1.f), _mm256_set1_ps(0.5f));
    __m256i iu = _mm256_cvttps_epi32(fu);
    __m256i iv = _mm256_cvttps_epi32(fv);
    __m256i index = _mm256_add_epi32(iu, _mm256_mullo_epi32(iv, _mm256_set1_epi32(uniform->cubemap->width)));
    __m256i color = _mm256_i32gather_epi32((int *)uniform->cubemap->data, _mm256_add_epi32(index, offset), 4);

    __m256 under = _mm256_loadu_ps((float *)((char *)uniform->depth + state->byte_offset));
    *allow = _mm256_cmp_ps(under, _mm256_setzero_ps(), _CMP_EQ_OQ);

    return color;
}

static const size_t uniform_offsets[] = {
    offsetof(uniforms, pixels),  offsetof(uniforms, depth),   offsetof(uniforms, id),
    offsetof(uniforms, texture), offsetof(uniforms, cubemap),
};

rohan_shader_desc get_description(void)
{
    return (rohan_shader_desc){.name = "s_skybox",
                               .render = rohan_render,
                               .instance_size = sizeof(uniforms),
                               .attribute_offset = offsetof(uniforms, dx),
                               .uniform_offsets = uniform_offsets};
}
