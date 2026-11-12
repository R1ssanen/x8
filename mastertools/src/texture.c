#include "texture.h"

#include <immintrin.h>

#include "logging.h"
#include "mtassert.h"
#include "types.h"
#include "utility/mtstring.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

mt_texture mt_texture_create(uint width, uint height, enum mt_texture_format format)
{
    uint stride = 0;
    switch (format)
    {
    case MT_TEXTURE_RGBA8888:
        stride = 4;
        break;
    case MT_TEXTURE_FLOAT32:
        stride = 4;
        break;
    default:
        LERROR("Unknown texture format '%d'", format);
        return (mt_texture){};
    }

    size_t pitch = width * stride;
    byte *data = _mm_malloc(pitch * height, 32);
    MT_ASSERT(data != NULL, "Could not allocate memory for texture");

    return (mt_texture){
        .pitch = pitch,
        .stride = stride,
        .width = width,
        .height = height,
        .log2_width = (uint)log2f(width),
        .data = data,
    };
}

void mt_texture_free(mt_texture *texture)
{
    if (texture)
    {
        _mm_free(texture->data);
    }
}

mt_texture mt_texture_load(mt_string_view path)
{
    int width, height, format;
    stbi_uc *data = stbi_load(path.str, &width, &height, &format, STBI_rgb_alpha);
    if (!data)
    {
        LERROR("Failed to load texture '%s'", path.str);
        return (mt_texture){}; // NULL;
    }

    return (mt_texture){
        .pitch = width * sizeof(int),
        .stride = sizeof(int),
        .width = width,
        .height = height,
        .log2_width = (uint)log2f(width),
        .data = data,
    };
}

void mt_texture_load_cubemap(mt_string_view path, mt_texture cubemap[6])
{
    cubemap[0] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/nx.png"));
    cubemap[1] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/px.png"));
    cubemap[2] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/ny.png"));
    cubemap[3] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/py.png"));
    cubemap[4] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/nz.png"));
    cubemap[5] = mt_texture_load(mt_string_refer_raw("resource/cubemap_1/pz.png"));

    size_t total_size = 0;
    for (size_t i = 0; i < 6; ++i)
    {
        total_size += cubemap[i].pitch * cubemap[i].height;
    }

    byte *new_data = _mm_malloc(total_size, 32);
    byte *ptr = new_data;

    for (size_t i = 0; i < 6; ++i)
    {
        memcpy(ptr, cubemap[i].data, cubemap[i].pitch * cubemap[i].height);
        // byte *tmp = cubemap[i].data;
        cubemap[i].data = ptr;
        ptr += cubemap[i].pitch * cubemap[i].height;
        // _mm_free(tmp);
    }
}

void mt_texture_clear(mt_texture *texture, uint clear)
{
    uint *data = (uint *)texture->data;
    for (size_t i = 0; i < (size_t)texture->width * (size_t)texture->height; ++i)
    {
        data[i] = clear;
    }
}
