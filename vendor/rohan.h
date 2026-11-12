#ifndef ROHAN_H
#define ROHAN_H

#ifndef __AVX2__
#error "rohan: AVX2 instructions required."
#endif

#if !defined(ROHAN_ATTRIB_COUNT) && defined(ROHAN_SHADER)
#define ROHAN_ATTRIB_COUNT 1 // to silence errors
#error "rohan: shader attribute count 'ROHAN_ATTRIB_COUNT' must be defined"
#endif

#if defined(_MSC_VER)
#ifdef ROHAN_BUILD
#define ROHAN_API __declspec(dllexport)
#else
#define ROHAN_API __declspec(dllimport)
#endif
#define ROHAN_ALIGNAS(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define ROHAN_API __attribute__((visibility("default")))
#define ROHAN_ALIGNAS(n) __attribute__((aligned(n)))
#else
#define ROHAN_API
#endif

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "texture.h"

typedef struct rohan_raster_state rohan_raster_state;
struct rohan_raster_state
{
    struct
    {
        __m256i x, y;
    } pos;

    __m256 z_under;
    __m256i mask;
    size_t byte_offset;
};

typedef struct rohan_shader_desc rohan_shader_desc;
struct rohan_shader_desc
{
    const char *name;
    void (*render)(rohan_shader_desc *restrict self, void *restrict instance, size_t target_pitch, size_t target_stride,
                   const float *restrict vertex0, const float *restrict vertex1, const float *restrict vertex2);
    size_t attribute_offset;
    size_t instance_size;
    size_t z_offset;
    const size_t *uniform_offsets;
};

typedef rohan_shader_desc (*rohan_get_description_fn)(void);

static inline void rohan_set_uniform_impl(const rohan_shader_desc *desc, void *restrict instance, size_t index,
                                          const void *restrict value, size_t stride)
{
    memcpy((char *)instance + desc->uniform_offsets[index], value, stride);
}

#define rohan_set_uniform(desc, instance, index, value_ptr)                                                            \
    rohan_set_uniform_impl(desc, instance, index, value_ptr, sizeof(*(value_ptr)))

// #define ROHAN_DEFINE_SHADER
#ifdef ROHAN_DEFINE_SHADER

#define swap(type, a, b)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        type tmp = a;                                                                                                  \
        a = b;                                                                                                         \
        b = tmp;                                                                                                       \
    } while (0)

static inline __m256 load_vector(const char *src)
{
    __m256 v;
    memcpy(&v, src, sizeof(__m256));
    return v;
}

static inline void store_vector(char *dest, __m256 v)
{
    memcpy(dest, &v, sizeof(__m256));
}

static const uint32_t ROHAN_ALIGNAS(32) tail_masks[][8] = {{0},
                                                           {-1, 0, 0, 0, 0, 0, 0, 0},
                                                           {-1, -1, 0, 0, 0, 0, 0, 0},
                                                           {-1, -1, -1, 0, 0, 0, 0, 0},
                                                           {-1, -1, -1, -1, 0, 0, 0, 0},
                                                           {-1, -1, -1, -1, -1, 0, 0, 0},
                                                           {-1, -1, -1, -1, -1, -1, 0, 0},
                                                           {-1, -1, -1, -1, -1, -1, -1, 0}};

static inline __m256i get_tail_mask(size_t n)
{
    return _mm256_load_si256((const __m256i *)tail_masks[n]);
}

#ifdef ROHAN_USE_DISCARD
static inline __m256i frag_main(void *restrict instance, const rohan_raster_state *restrict state,
                                __m256 *restrict allow);
#else
static inline __m256i frag_main(void *restrict instance, const rohan_raster_state *restrict state);
#endif

static inline void tail_no_depth(char *pixels, void *restrict instance, rohan_raster_state *restrict state,
                                 __m256i tail_mask)
{
#ifdef ROHAN_USE_DISCARD
    __m256 allow;
    __m256i frag = frag_main(instance, state, &allow);
    tail_mask = _mm256_and_si256(_mm256_castps_si256(allow), tail_mask);
#else
    __m256i frag = frag_main(instance, state);
#endif
    _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), tail_mask, frag);
}

static inline void tail_with_depth(char *pixels, char *depth, char *z_loc, void *restrict instance,
                                   rohan_raster_state *restrict state, __m256i tail_mask)
{
    __m256 z = load_vector(z_loc);
    state->z_under = _mm256_loadu_ps((float *)(depth + state->byte_offset));

#ifdef ROHAN_DEPTH_TEST_LESS_THAN
    __m256 z_mask = _mm256_cmp_ps(z, state->z_under, _CMP_LT_OQ);
#else
    __m256 z_mask = _mm256_cmp_ps(z, state->z_under, _CMP_GT_OQ);
#endif

    __m256 mask = _mm256_and_ps(_mm256_castsi256_ps(tail_mask), z_mask);
    if (_mm256_testz_ps(mask, mask) == 1)
    {
        return;
    }

#ifdef ROHAN_USE_DISCARD
    __m256 allow;
    __m256i frag = frag_main(instance, state, &allow);
    mask = _mm256_and_ps(allow, mask);
#else
    __m256i frag = frag_main(instance, state);
#endif
    _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), _mm256_castps_si256(mask), frag);
    _mm256_storeu_ps((float *)(depth + state->byte_offset), _mm256_blendv_ps(state->z_under, z, mask));
}

static inline void body_no_depth(char *pixels, void *restrict instance, rohan_raster_state *restrict state)
{
#ifdef ROHAN_USE_DISCARD
    __m256 allow;
    __m256i frag = frag_main(instance, state, &allow);
    _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), _mm256_castps_si256(allow), frag);
#else
    __m256i frag = frag_main(instance, state);
    _mm256_storeu_si256((__m256i_u *)(pixels + state->byte_offset), frag);
#endif
}

static inline void body_with_depth(char *pixels, char *depth, char *z_loc, void *restrict instance,
                                   rohan_raster_state *restrict state)
{
    __m256i frag;
    __m256 z = load_vector(z_loc);
    state->z_under = _mm256_loadu_ps((float *)(depth + state->byte_offset));

#ifdef ROHAN_DEPTH_TEST_LESS_THAN
    __m256 z_mask = _mm256_cmp_ps(z, state->z_under, _CMP_LT_OQ);
#else
    __m256 z_mask = _mm256_cmp_ps(z, state->z_under, _CMP_GT_OQ);
#endif

#ifdef ROHAN_USE_DISCARD
    __m256 allow;
#endif

    switch (_mm256_movemask_ps(z_mask))
    {
    case 0:
        break;
    case 0xff:
#ifdef ROHAN_USE_DISCARD
        frag = frag_main(instance, state, &allow);
        _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), _mm256_castps_si256(allow), frag);
        _mm256_storeu_ps((float *)(depth + state->byte_offset), _mm256_blendv_ps(state->z_under, z, allow));
#else
        frag = frag_main(instance, state);
        _mm256_storeu_si256((__m256i_u *)(pixels + state->byte_offset), frag);
        _mm256_storeu_ps((float *)(depth + state->byte_offset), z);
#endif
        break;
    default:
#ifdef ROHAN_USE_DISCARD
        frag = frag_main(instance, state, &allow);
        allow = _mm256_and_ps(allow, z_mask);
        _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), _mm256_castps_si256(allow), frag);
        _mm256_storeu_ps((float *)(depth + state->byte_offset), _mm256_blendv_ps(state->z_under, z, allow));
#else
        frag = frag_main(instance, state);
        _mm256_maskstore_epi32((int *)(pixels + state->byte_offset), _mm256_castps_si256(z_mask), frag);
        _mm256_storeu_ps((float *)(depth + state->byte_offset), _mm256_blendv_ps(state->z_under, z, z_mask));
#endif
        break;
    }
}

static inline void rasterize(rohan_shader_desc *desc, const float attr_dx[static restrict ROHAN_ATTRIB_COUNT],
                             const float attr_dy[static restrict ROHAN_ATTRIB_COUNT],
                             const __m256 attr_p0[static restrict ROHAN_ATTRIB_COUNT], void *restrict instance,
                             size_t target_pitch, size_t target_stride, float x_left, float dx_dy_left, float x_right,
                             float dx_dy_right, int y_top, int y_bottom)
{
    rohan_raster_state state = {0};
    char *attributes = (char *)instance + desc->attribute_offset;

    mt_texture pixel_tex;
    memcpy(&pixel_tex, (char *)instance + desc->uniform_offsets[0], sizeof(pixel_tex));
    char *pixels = (char *)pixel_tex.data;

#if !defined(ROHAN_NO_DEPTH)
    char *depth;
    memcpy(&depth, (char *)instance + desc->uniform_offsets[1], sizeof(depth));
    char *z_loc = (char *)instance + desc->z_offset;
#endif

    // int y0 = fmaxf(ceilf(y_top - 0.5f), 0.f);
    // int y1 = fminf(ceilf(y_bottom - 0.5f), pixel_tex.height);
    // if (y1 <= y0)
    // {
    //     return;
    // }

    size_t row_byte_offset = y_top * target_pitch;
    for (int dy = 0; dy < y_bottom - y_top; dy += 1, row_byte_offset += target_pitch)
    {
        // float sample_offset = (float)y0 + (float)dy + 0.5f - y_top;

        float left = fmaf(dx_dy_left, dy, x_left);
        float right = fmaf(dx_dy_right, dy, x_right);

        // int x0 = fmaxf(ceilf(left - 0.5f), 0);
        // int x1 = fminf(ceilf(right - 0.5f), pixel_tex.width);
        int x0 = left;
        int x1 = right;

        int diff = x1 - x0;
        if (diff <= 0)
            continue;

        // float sample_x_offset = (float)x0 + 0.5f - left;

        state.byte_offset = row_byte_offset + x0 * target_stride;
        state.pos.x = _mm256_add_epi32(_mm256_set1_epi32(x0), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        state.pos.y = _mm256_set1_epi32(y_top + dy);

        for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
        {
            __m256 attr = _mm256_fmadd_ps(_mm256_set1_ps(dy), _mm256_set1_ps(attr_dy[i]), attr_p0[i]);
            // attr = _mm256_fmadd_ps(_mm256_set1_ps(sample_x_offset), _mm256_set1_ps(attr_dx[i] * 0.125f), attr);
            store_vector(attributes + i * sizeof(__m256), attr);
        }

        for (int i = 0; i < diff / 8; ++i)
        {
#ifdef ROHAN_NO_DEPTH
            body_no_depth(pixels, instance, &state);
#else
            body_with_depth(pixels, depth, z_loc, instance, &state);
#endif

            state.pos.x = _mm256_add_epi32(state.pos.x, _mm256_set1_epi32(8));
            state.byte_offset += target_stride * 8;

            for (size_t j = 0; j < ROHAN_ATTRIB_COUNT; ++j)
            {
                char *field = attributes + j * sizeof(__m256);
                __m256 attr = load_vector(field);
                attr = _mm256_add_ps(attr, _mm256_set1_ps(attr_dx[j]));
                store_vector(field, attr);
            }
        }

        size_t tail = diff % 8;
        if (tail == 0)
        {
            continue;
        }

        __m256i tail_mask = get_tail_mask(tail);
        state.pos.x = _mm256_blendv_epi8(_mm256_setzero_si256(), state.pos.x, tail_mask);

#ifdef ROHAN_NO_DEPTH
        tail_no_depth(pixels, instance, &state, tail_mask);
#else
        tail_with_depth(pixels, depth, z_loc, instance, &state, tail_mask);
#endif
    }
}

static inline void rohan_render(rohan_shader_desc *restrict desc, void *restrict instance, size_t target_pitch,
                                size_t target_stride, const float *restrict vertex0, const float *restrict vertex1,
                                const float *restrict vertex2)
{
    float x0 = truncf(vertex0[0]), x1 = truncf(vertex1[0]), x2 = truncf(vertex2[0]);
    float y0 = truncf(vertex0[1]), y1 = truncf(vertex1[1]), y2 = truncf(vertex2[1]);
    const float *attr_0 = vertex0 + 2, *attr_1 = vertex1 + 2, *attr_2 = vertex2 + 2;

    if (y0 > y1)
    {
        swap(float, x0, x1);
        swap(float, y0, y1);
        swap(const float *, attr_0, attr_1);
    }
    if (y1 > y2)
    {
        swap(float, x1, x2);
        swap(float, y1, y2);
        swap(const float *, attr_1, attr_2);
    }
    if (y0 > y1)
    {
        swap(float, x0, x1);
        swap(float, y0, y1);
        swap(const float *, attr_0, attr_1);
    }

    float attr_dx[ROHAN_ATTRIB_COUNT];
    float attr_dy[ROHAN_ATTRIB_COUNT];
    __m256 attr_p0[ROHAN_ATTRIB_COUNT];

    float inv_dy_ac = 1.f / (y2 - y0);
    const __m256 seq = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

    if (y0 == y1) // flat top
    {
        float dxdy_left = (x2 - x0) * inv_dy_ac;
        float dxdy_right = (x2 - x1) * inv_dy_ac;

        if (x0 < x1)
        {
            float inv_dx_ab = 1.f / (x1 - x0);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                // desc->_attr_dx[i] = _mm256_set1_ps((attr_1[i] - attr_0[i]) * inv_dx_ab * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_2[i] - attr_0[i]) * inv_dy_ac);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_0[i]);

                float dx = (attr_1[i] - attr_0[i]) * inv_dx_ab;
                attr_dx[i] = dx * 8.f;
                attr_dy[i] = (attr_2[i] - attr_0[i]) * inv_dy_ac;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_0[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x0, dxdy_left, x1,
                      dxdy_right, y0, y2);
        }
        else
        {
            float inv_dy_bc = 1.f / (y2 - y1);
            float inv_dx_ab = 1.f / (x0 - x1);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                // desc->_attr_dx[i] = _mm256_set1_ps((attr_0[i] - attr_1[i]) * -inv_dx_ab * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_2[i] - attr_1[i]) * inv_dy_bc);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_1[i]);

                float dx = (attr_0[i] - attr_1[i]) * inv_dx_ab;
                attr_dx[i] = dx * 8.f;
                attr_dy[i] = (attr_2[i] - attr_1[i]) * inv_dy_bc;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_1[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x1, dxdy_right, x0,
                      dxdy_left, y0, y2);
        }
    }

    else if (y1 == y2) // flat bottom
    {
        float dxdy_left = (x1 - x0) * inv_dy_ac;
        float dxdy_right = (x2 - x0) * inv_dy_ac;

        if (x1 < x2)
        {
            float inv_dy_ab = 1.f / (y1 - y0);
            float inv_dx_bc = 1.f / (x2 - x1);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                // desc->_attr_dx[i] = _mm256_set1_ps((attr_2[i] - attr_1[i]) * inv_dx_bc * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_1[i] - attr_0[i]) * inv_dy_ab);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_0[i]);

                float dx = (attr_2[i] - attr_1[i]) * inv_dx_bc;
                attr_dx[i] = dx * 8.f;
                attr_dy[i] = (attr_1[i] - attr_0[i]) * inv_dy_ab;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_0[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x0, dxdy_left, x0,
                      dxdy_right, y0, y2);
        }
        else
        {
            float inv_dx_bc = 1.f / (x1 - x2);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                // desc->_attr_dx[i] = _mm256_set1_ps((attr_1[i] - attr_2[i]) * -inv_dx_bc * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_2[i] - attr_0[i]) * inv_dy_ac);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_0[i]);

                float dx = (attr_1[i] - attr_2[i]) * inv_dx_bc;
                attr_dx[i] = dx * 8.f;
                attr_dy[i] = (attr_2[i] - attr_0[i]) * inv_dy_ac;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_0[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x0, dxdy_right, x0,
                      dxdy_left, y0, y2);
        }
    }

    else
    {
        float dy_ab = y1 - y0;
        float dxdy_ac = (x2 - x0) * inv_dy_ac;
        float x3 = fmaf(dxdy_ac, dy_ab, x0);

        float inv_dy_ab = 1.f / dy_ab;
        float inv_dy_bc = 1.f / (y2 - y1);

        float dxdy_top = (x1 - x0) * inv_dy_ab;
        float dxdy_bottom = (x2 - x1) * inv_dy_bc;

        if (x1 < x3) // long side right
        {
            float dx[ROHAN_ATTRIB_COUNT];
            float inv_dx_bd = 1.f / (x3 - x1);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                float attr_3 = fmaf((attr_2[i] - attr_0[i]) * inv_dy_ac, dy_ab, attr_0[i]);

                // desc->_attr_dx[i] = _mm256_set1_ps((attr_3 - attr_1[i]) * inv_dx_bd * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_1[i] - attr_0[i]) * inv_dy_ab);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_0[i]);

                dx[i] = (attr_3 - attr_1[i]) * inv_dx_bd;
                attr_dx[i] = dx[i] * 8.f;
                attr_dy[i] = (attr_1[i] - attr_0[i]) * inv_dy_ab;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx[i]), seq, _mm256_set1_ps(attr_0[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x0, dxdy_top, x0, dxdy_ac,
                      y0, y1);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_2[i] - attr_1[i]) * inv_dy_bc);
                // desc->_attr_0[i] = _mm256_set1_ps(attr_1[i]);

                attr_dy[i] = (attr_2[i] - attr_1[i]) * inv_dy_bc;
                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx[i]), seq, _mm256_set1_ps(attr_1[i]));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x1, dxdy_bottom, x3,
                      dxdy_ac, y1, y2);
        }
        else // long side left
        {
            __m256 attr_p01[ROHAN_ATTRIB_COUNT];
            float inv_dx_bd = 1.f / (x1 - x3);

            for (size_t i = 0; i < ROHAN_ATTRIB_COUNT; ++i)
            {
                float attr_3 = fmaf((attr_2[i] - attr_0[i]) * inv_dy_ac, dy_ab, attr_0[i]);

                // desc->_attr_dx[i] = _mm256_set1_ps((attr_1[i] - attr_3) * -inv_dx_bd * 8.f);
                // desc->_attr_dy[i] = _mm256_set1_ps((attr_3 - attr_0[i]) * inv_dy_ab);

                // desc->_attr_0[i] = _mm256_set1_ps(attr_0[i]);
                // desc->_attr_01[i] = _mm256_set1_ps(attr_3);

                float dx = (attr_1[i] - attr_3) * inv_dx_bd;
                attr_dx[i] = dx * 8.f;
                attr_dy[i] = (attr_3 - attr_0[i]) * inv_dy_ab;

                attr_p0[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_0[i]));
                attr_p01[i] = _mm256_fmadd_ps(_mm256_set1_ps(dx), seq, _mm256_set1_ps(attr_3));
            }

            rasterize(desc, attr_dx, attr_dy, attr_p0, instance, target_pitch, target_stride, x0, dxdy_ac, x0, dxdy_top,
                      y0, y1);
            rasterize(desc, attr_dx, attr_dy, attr_p01, instance, target_pitch, target_stride, x3, dxdy_ac, x1,
                      dxdy_bottom, y1, y2);
        }
    }
}

#endif

#endif