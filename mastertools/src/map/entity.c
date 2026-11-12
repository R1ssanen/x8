#include "map/entity.h"

#include <immintrin.h>
#include <string.h>

#include "cJSON.h"
#include "loaders/wavefront_obj.h"
#include "logging.h"
#include "system/library.h"
#include "texture.h"
#include "types.h"
#include "utility/file.h"

static const byte default_tex_data[] = {0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00, 0x00,
                                        0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff,
                                        0x00, 0xff, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00,
                                        0xff, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
                                        0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00};

void mt_entity_free(mt_entity *entity)
{
    if (!entity)
    {
        return;
    }

    mt_array_foreach(&entity->meshes, mt_mesh, mesh)
    {
        mt_mesh_free(mesh);
    }

    mt_array_free(&entity->meshes);

    _mm_free(entity->shader_instance);

    mt_library_free(&entity->shader_lib);
    _mm_free(entity);
}

mt_entity *parse_node_entity_json(const cJSON *object)
{
    const cJSON *file = cJSON_GetObjectItemCaseSensitive(object, "file");
    if (!file || !cJSON_IsString(file))
    {
        LERROR("Entity must have a valid file associated");
        return NULL;
    }

    mt_string_view path = mt_string_refer_raw(file->valuestring);

    mt_string extension;
    if (!mt_file_get_extension(path, &extension))
    {
        LERROR("No extension");
        return NULL;
    }

    if (mt_string_equal_sr(extension, ".obj"))
    {
        mt_string_free(&extension);
        LINFO("Loading OBJ file '%s'", path.str);
    }
    else
    {
        mt_string_free(&extension);
        LERROR("Unsupported object file format");
        return NULL;
    }

    mt_entity *entity = _mm_malloc(sizeof(mt_entity), 32);
    if (!mt_load_wavefront_obj(path, entity))
    {
        return NULL;
    }

    cJSON *shader = cJSON_GetObjectItemCaseSensitive(object, "shader");
    if (!shader)
    {
        LERROR("Entity must have a shader");
        return NULL;
    }

    mt_string_view shader_path = mt_string_refer_raw(shader->valuestring);
    if (!mt_library_load(shader_path, &entity->shader_lib))
    {
        LERROR("Failed to load shader library '%s'", shader_path.str);
    }

    void *get_spec = mt_library_load_symbol(&entity->shader_lib, mt_string_refer_raw("get_description"));
    entity->desc = ((rohan_get_description_fn)get_spec)();
    entity->shader_instance = _mm_malloc(entity->desc.instance_size, 32);

    cJSON *layer = cJSON_GetObjectItemCaseSensitive(object, "layer");
    if (layer && cJSON_IsNumber(layer))
    {
        if (layer->valueint > 3)
        {
            LWARN("Entity layer exceeds maximum value of 3");
            entity->depth_layer = 3;
        }
        else
        {
            entity->depth_layer = layer->valueint;
        }
    }
    else
    {
        entity->depth_layer = 0;
    }

    cJSON *texture = cJSON_GetObjectItemCaseSensitive(object, "texture");
    if (!texture)
    {
        entity->texture = (mt_texture){
            .width = 4,
            .height = 4,
            .pitch = 4 * sizeof(int),
            .stride = sizeof(int),
            .log2_width = 2,
            .data = default_tex_data,
        };
    }
    else
    {
        entity->texture = mt_texture_load(mt_string_refer_raw(texture->valuestring));
    }

    return entity;
}
