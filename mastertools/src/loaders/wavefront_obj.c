#include "loaders/wavefront_obj.h"

#include <math.h>
#include <stdlib.h>

#include "logging.h"
#include "map/entity.h"
#include "types.h"
#include "utility/array.h"
#include "utility/file.h"
#include "utility/hash.h"
#include "utility/mtstring.h"

static inline bool char_within_str(char c, const char *str)
{
    for (; *str != '\0'; ++str)
    {
        if (c == *str)
        {
            return true;
        }
    }

    return false;
}

static char *trim_leading(char *str, const char *whitespace)
{
    char *it = str;
    for (; *it != '\0'; ++it)
    {
        if (!char_within_str(*it, whitespace))
        {
            break;
        }
    }

    return it;
}

typedef struct tokenizer tokenizer;
struct tokenizer
{
    char *src;
    size_t offset;
    char delimiter;
    bool ended;
};

static inline uint make_id(size_t base_hash, uint face)
{
    size_t v = base_hash ^ (size_t)face * 0x9e3779b97f4a7c15ULL;
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    return (uint)v;
}

static inline char *next(tokenizer *tokenizer)
{
    if (tokenizer->ended)
    {
        return NULL;
    }

    size_t start = tokenizer->offset;
    for (size_t i = start; tokenizer->src[i] != '\0'; ++i)
    {
        if (tokenizer->src[i] == tokenizer->delimiter)
        {
            tokenizer->src[i] = '\0';
            tokenizer->offset = i + 1;
            return trim_leading(tokenizer->src + start, " \t\v\f\r");
        }
    }

    tokenizer->ended = true;
    return trim_leading(tokenizer->src + start, " \t\v\f\r");
}

bool mt_load_wavefront_obj(mt_string_view path, mt_entity *entity)
{
    char *buffer;
    size_t len;
    if (!mt_file_read(path, &buffer, &len))
    {
        LERROR("Could not open .obj file '%s'", path.str);
        return NULL;
    }

    entity->meshes = mt_array_create(sizeof(mt_mesh));
    mt_mesh *mesh = NULL;
    size_t base_hash = mt_hash_fnv_1a_64(path);

    mesh = mt_array_push(&entity->meshes, NULL);
    mesh->vertices = mt_array_create(sizeof(float));
    mesh->uvs = mt_array_create(sizeof(float));
    mesh->indices = mt_array_create(sizeof(uint));

    uint counter = 1;
    tokenizer line_tok = {.src = buffer, .offset = 0, .delimiter = '\n'};

    for (char *line; (line = next(&line_tok));)
    {
        if (false && line[0] == 'o')
        {
            mesh = mt_array_push(&entity->meshes, NULL);
            mesh->vertices = mt_array_create(sizeof(float));
            mesh->uvs = mt_array_create(sizeof(float));
            mesh->indices = mt_array_create(sizeof(uint));
        }

        else if ((line[0] == 'v') && (line[1] == ' '))
        {
            tokenizer vertex_tok = {.src = trim_leading(line, " v\t\v\f\r"), .offset = 0, .delimiter = ' '};

            char *vertex = next(&vertex_tok);
            float x = strtof(vertex, NULL);
            mt_array_push(&mesh->vertices, &x);

            vertex = next(&vertex_tok);
            float y = strtof(vertex, NULL);
            mt_array_push(&mesh->vertices, &y);

            vertex = next(&vertex_tok);
            float z = strtof(vertex, NULL);
            mt_array_push(&mesh->vertices, &z);
        }

        else if ((line[0] == 'v') && (line[1] == 't'))
        {
            tokenizer uv_tok = {.src = trim_leading(line, " vt\t\v\f\r"), .offset = 0, .delimiter = ' '};

            char *uv = next(&uv_tok);
            float u = strtof(uv, NULL);
            mt_array_push(&mesh->uvs, &u);

            uv = next(&uv_tok);
            float v = -strtof(uv, NULL);
            // if (v >= 0.f && v <= 1.f) // flip to match +y up convention
            // {
            //     v = 1.f - v;
            // }
            // else
            // {
            //     v = -v;
            // }

            mt_array_push(&mesh->uvs, &v);
        }

        else if (line[0] == 'f')
        {
            uint id = make_id(base_hash, counter++);
            mt_array_push(&mesh->indices, &id);

            tokenizer face_tok = {.src = trim_leading(line, " f\t\v\f\r"), .offset = 0, .delimiter = ' '};

            for (char *vertex; (vertex = next(&face_tok));)
            {
                tokenizer index_tok = {.src = trim_leading(vertex, " \t\v\f\r"), .offset = 0, .delimiter = '/'};

                char *id = next(&index_tok);
                int vertex_id = strtoul(id, NULL, 10) - 1;
                mt_array_push(&mesh->indices, &vertex_id);

                id = next(&index_tok);
                if (id)
                {
                    int uv_id = strtoul(id, NULL, 10) - 1;
                    mt_array_push(&mesh->indices, &uv_id);
                }
                else
                {
                    int zero = 0;
                    mt_array_push(&mesh->indices, &zero);
                }
            }
        }
    }

    free(buffer);
    return true;
}
