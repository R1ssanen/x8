#define ROHAN_DEFINE_SHADER
#define ROHAN_ATTRIB_COUNT 3
// #define ROHAN_USE_DISCARD
#include "rohan.h"

typedef struct uniforms uniforms;
struct uniforms
{
    __m256 q;
    __m256 u;
    __m256 v;

    mt_texture texture;
    mt_texture pixels;
    float *depth;
    uint32_t id;
    mt_texture *cubemap;

    __m256 lx, ly, lz;
    __m256 nx, ny, nz;
    __m256 cx, cy, cz;
};

static inline __m256 clamp_ps(__m256 x, float min, float max)
{
    return _mm256_max_ps(_mm256_set1_ps(min), _mm256_min_ps(x, _mm256_set1_ps(max)));
}

static inline __m256i sample_2d(__m256 u, __m256 v, const mt_texture *texture)
{
    // __m256 fu = _mm256_mul_ps(u, _mm256_set1_ps(texture->width));
    // __m256 fv = _mm256_mul_ps(v, _mm256_set1_ps(texture->height));

    // __m256i iu = _mm256_and_si256(_mm256_cvttps_epi32(fu), _mm256_set1_epi32(texture->width - 1));
    // __m256i iv = _mm256_and_si256(_mm256_cvttps_epi32(fv), _mm256_set1_epi32(texture->height - 1));

    // u = clamp_ps(u, 0.f, 1.f);
    // v = clamp_ps(v, 0.f, 1.f);

    u = _mm256_sub_ps(u, _mm256_floor_ps(u));
    v = _mm256_sub_ps(v, _mm256_floor_ps(v));

    __m256 fu = _mm256_fmadd_ps(u, _mm256_set1_ps(texture->width - 1.f), _mm256_set1_ps(0.5f));
    __m256 fv = _mm256_fmadd_ps(v, _mm256_set1_ps(texture->height - 1.f), _mm256_set1_ps(0.5f));

    // fu = _mm256_min_ps(fu, _mm256_set1_ps(texture->width - 1.f));
    // fv = _mm256_min_ps(fv, _mm256_set1_ps(texture->height - 1.f));
    // fu = clamp_ps(fu, 0, texture->width - 1.f);
    // fv = clamp_ps(fv, 0, texture->height - 1.f);

    __m256i iu = _mm256_cvttps_epi32(fu);
    __m256i iv = _mm256_cvttps_epi32(fv);

    //__m256i index = _mm256_add_epi32(iu, _mm256_slli_epi32(iv, texture->log2_width));
    __m256i index = _mm256_add_epi32(iu, _mm256_mullo_epi32(iv, _mm256_set1_epi32(texture->width)));
    return _mm256_i32gather_epi32((int *)texture->data, index, 4);
}

static inline __m256 abs_ps(__m256 x)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.f), x);
}

static inline __m256 wrap_ps(__m256 x)
{
    return _mm256_sub_ps(x, _mm256_floor_ps(x));
}

#include <stdio.h>

__m256i scale_i32(__m256i pixel, __m256i t)
{
    const __m256i mask = _mm256_set1_epi32(0xff);
    __m256i r = _mm256_and_si256(pixel, mask);
    __m256i g = _mm256_and_si256(_mm256_srli_epi32(pixel, 8), mask);
    __m256i b = _mm256_and_si256(_mm256_srli_epi32(pixel, 16), mask);

    r = _mm256_srli_epi32(_mm256_mullo_epi32(r, t), 8);
    g = _mm256_mullo_epi32(g, t);
    b = _mm256_srli_epi32(_mm256_mullo_epi32(b, t), 8);

    __m256i frag = _mm256_or_si256(r, _mm256_andnot_si256(mask, g));
    return _mm256_or_si256(frag, _mm256_slli_epi32(b, 16));
}

// struct g_attributes
// {
//     int _;
// };

typedef enum attrib_kind
{
    ATTR_FLOAT3,
    ATTR_INT3,
    ATTR_FLOAT,
    ATTR_INT,
} attrib_kind;

typedef struct attrib attrib;
struct attrib
{
    attrib_kind kind;
};

static const attrib s_attrib_desc[ROHAN_ATTRIB_COUNT] = {
    {ATTR_FLOAT3},
    {ATTR_INT},
};

__m256i frag_main(void *restrict instance, const rohan_raster_state *restrict state)
{
    uniforms *uniform = instance;
    __m256 z = _mm256_rcp_ps(uniform->q);

    // __m256 nz = _mm256_sub_ps(z, _mm256_set1_ps(0.1f));
    // nz = _mm256_div_ps(z, _mm256_sub_ps(_mm256_set1_ps(10.f), _mm256_set1_ps(0.1f)));

    // *allow = _mm256_set1_ps(-0.f);
    // return _mm256_cvttps_epi32(_mm256_mul_ps(nz, _mm256_set1_ps(255.f)));

    __m256 u = _mm256_mul_ps(uniform->u, z);
    __m256 v = _mm256_mul_ps(uniform->v, z);

    __m256i texel = sample_2d(u, v, &uniform->texture);
    // return texel;

    const __m256i mask = _mm256_set1_epi32(0xff);
    __m256i r = _mm256_and_si256(texel, mask);
    __m256i g = _mm256_and_si256(_mm256_srli_epi32(texel, 8), mask);

    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 inv_half = _mm256_set1_ps(1.f / 127.5f);

    __m256 du = _mm256_fmsub_ps(_mm256_cvtepi32_ps(r), inv_half, one);
    __m256 dv = _mm256_fmsub_ps(_mm256_cvtepi32_ps(g), inv_half, one);

    __m256 inv_w = _mm256_rcp_ps(_mm256_set1_ps(uniform->pixels.width));
    __m256 inv_h = _mm256_rcp_ps(_mm256_set1_ps(uniform->pixels.height));

    const __m256 half = _mm256_set1_ps(0.f);
    __m256 su = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(state->pos.x), half), inv_w);
    __m256 sv = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(state->pos.y), half), inv_h);

    const __m256 intensity = _mm256_set1_ps(0.1f);
    su = clamp_ps(_mm256_fmadd_ps(du, intensity, su), 0.f, 1.f);
    sv = clamp_ps(_mm256_fmadd_ps(dv, intensity, sv), 0.f, 1.f);

    __m256i sx = _mm256_cvttps_epi32(_mm256_mul_ps(su, _mm256_set1_ps(uniform->pixels.width - 1)));
    __m256i sy = _mm256_cvttps_epi32(_mm256_mul_ps(sv, _mm256_set1_ps(uniform->pixels.height - 1)));

    // sx = _mm256_min_epi32(sx, _mm256_set1_epi32(uniform->pixels.width - 1));
    // sy = _mm256_min_epi32(sy, _mm256_set1_epi32(uniform->pixels.height - 1));

    __m256i index = _mm256_add_epi32(sx, _mm256_mullo_epi32(sy, _mm256_set1_epi32(uniform->pixels.width)));
    __m256i sample = _mm256_i32gather_epi32((int *)uniform->pixels.data, index, 4);
    return scale_i32(sample, _mm256_set1_epi32(200));
}

static const size_t uniform_offsets[] = {
    offsetof(uniforms, pixels),  offsetof(uniforms, depth), offsetof(uniforms, id), offsetof(uniforms, texture),
    offsetof(uniforms, cubemap), offsetof(uniforms, lx),    offsetof(uniforms, ly), offsetof(uniforms, lz),
    offsetof(uniforms, nx),      offsetof(uniforms, ny),    offsetof(uniforms, nz),
};

rohan_shader_desc get_description(void)
{
    return (rohan_shader_desc){.name = "s_refraction",
                               .render = rohan_render,
                               .instance_size = sizeof(uniforms),
                               .attribute_offset = offsetof(uniforms, q),
                               .uniform_offsets = uniform_offsets,
                               .z_offset = offsetof(uniforms, q)};
}
