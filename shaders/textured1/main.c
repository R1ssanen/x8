#define ROHAN_DEFINE_SHADER
#define ROHAN_ATTRIB_COUNT 6
#include "rohan.h"

typedef struct uniforms uniforms;
struct uniforms
{
    __m256 q;
    __m256 u;
    __m256 v;

    __m256 wx, wy, wz;

    mt_texture texture;
    mt_texture pixels;
    float *depth;
    uint32_t id;
    mt_texture *cubemap;

    __m256 lx, ly, lz;
    __m256 nx, ny, nz;
    __m256 cx, cy, cz;
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

static inline __m256i sample_2d_zo(__m256 u, __m256 v, const mt_texture *texture)
{
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256i iu = _mm256_cvttps_epi32(_mm256_fmadd_ps(u, _mm256_set1_ps(texture->width - 1.f), half));
    __m256i iv = _mm256_cvttps_epi32(_mm256_fmadd_ps(v, _mm256_set1_ps(texture->height - 1.f), half));

    __m256i index = _mm256_add_epi32(iu, _mm256_mullo_epi32(iv, _mm256_set1_epi32(texture->width)));
    return _mm256_i32gather_epi32((int *)texture->data, index, 4);
}

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

__m256i frag_main(void *restrict instance, const rohan_raster_state *restrict state)
{
    uniforms *uniform = instance;
    __m256 z = _mm256_rcp_ps(uniform->q);

    __m256 u = _mm256_mul_ps(uniform->u, z);
    __m256 v = _mm256_mul_ps(uniform->v, z);

    u = _mm256_sub_ps(u, _mm256_floor_ps(u));
    v = _mm256_sub_ps(v, _mm256_floor_ps(v));

    __m256i texel = sample_2d_zo(u, v, &uniform->texture);
    // return texel;

    __m256 dx = _mm256_mul_ps(uniform->wx, z);
    __m256 dy = _mm256_mul_ps(uniform->wy, z);
    __m256 dz = _mm256_mul_ps(uniform->wz, z);

    __m256 inv_d = _mm256_rsqrt_ps(_mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz))));
    __m256 inv_d2 = _mm256_mul_ps(inv_d, inv_d);

    __m256 NdotD = _mm256_fmadd_ps(uniform->nx, dx, _mm256_fmadd_ps(uniform->ny, dy, _mm256_mul_ps(uniform->nz, dz)));
    __m256 L = _mm256_max_ps(NdotD, _mm256_setzero_ps());

    const __m256 ambient = _mm256_set1_ps(0.1f);
    const __m256 intensity = _mm256_set1_ps(1.5f);
    __m256 atten = _mm256_min_ps(_mm256_fmadd_ps(_mm256_mul_ps(inv_d2, intensity), L, ambient), _mm256_set1_ps(1.f));
    __m256i atteni = _mm256_cvttps_epi32(_mm256_mul_ps(atten, _mm256_set1_ps(255.f)));

    __m256i color = scale_i32(texel, atteni);
    return color;
}

static const size_t uniform_offsets[] = {
    offsetof(uniforms, pixels),  offsetof(uniforms, depth), offsetof(uniforms, id), offsetof(uniforms, texture),
    offsetof(uniforms, cubemap), offsetof(uniforms, lx),    offsetof(uniforms, ly), offsetof(uniforms, lz),
    offsetof(uniforms, nx),      offsetof(uniforms, ny),    offsetof(uniforms, nz),
};

rohan_shader_desc get_description(void)
{
    return (rohan_shader_desc){.name = "s_textured1",
                               .render = rohan_render,
                               .instance_size = sizeof(uniforms),
                               .attribute_offset = offsetof(uniforms, q),
                               .uniform_offsets = uniform_offsets,
                               .z_offset = offsetof(uniforms, q)};
}
