#ifndef MASTERTOOLS_CAMERA_H_
#define MASTERTOOLS_CAMERA_H_

#include "math/common.h"
#include "math/vec3.h"
#include "types.h"

struct cJSON;
struct mt_map;

typedef struct mt_camera mt_camera;
struct mt_camera
{
    mvec3 center;
    mvec3 pos;
    mvec3 angle;
    float near;
    float far;
    float r_fov;
    bool active;
};

mt_camera *parse_node_camera_json(struct cJSON *object, struct mt_map *map);

mvec3 mt_camera_get_forward(mt_camera *camera);

mmat4 mt_camera_get_view_projection(mt_camera *camera, float aspect);

#endif