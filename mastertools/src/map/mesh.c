#include "map/mesh.h"

#include <string.h>

#include "utility/zero_init.h"

void mt_mesh_free(mt_mesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    mt_array_free(&mesh->vertices);
    mt_array_free(&mesh->uvs);
    mt_array_free(&mesh->indices);

    MT_ZERO_INIT(mesh);
}
