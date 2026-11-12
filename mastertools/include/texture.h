#ifndef MASTERTOOLS_TEXTURE_H_
#define MASTERTOOLS_TEXTURE_H_

#include "types.h"
#include "utility/array.h"
#include "utility/mtstring.h"

enum mt_texture_format
{
    MT_TEXTURE_RGBA8888,
    MT_TEXTURE_FLOAT32,
};

typedef struct mt_texture mt_texture;
struct mt_texture
{
    size_t pitch;
    uint width;
    uint height;
    uint stride;
    uint log2_width;
    byte *data;
};

mt_texture mt_texture_create(uint w, uint h, enum mt_texture_format format);

void mt_texture_free(mt_texture *texture);

mt_texture mt_texture_load(mt_string_view path);

void mt_texture_load_cubemap(mt_string_view path, mt_texture cubemap[6]);

void mt_texture_clear(mt_texture *texture, uint clear);

#endif