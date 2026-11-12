#include "map/camera.h"

#include "cJSON.h"
#include "map/map.h"
#include "math/constants.h"
#include "math/vec2.h"
#include "math/vec3.h"

mt_camera *parse_node_camera_json(struct cJSON *object, mt_map *map)
{
    int camera_id = map->cameras.size;
    mt_camera *camera = mt_array_push(&map->cameras, NULL);

    camera->center = mt_vec3(0.f, 0.f, 0.f);

    cJSON *active = cJSON_GetObjectItemCaseSensitive(object, "active");
    if (active && (active->valueint != 0))
    {
        map->active_camera_id = camera_id;
    }

    cJSON *pos = cJSON_GetObjectItemCaseSensitive(object, "pos");
    if (!pos || !cJSON_IsArray(pos))
    {
        camera->pos = mt_vec3_zero();
    }
    else
    {
        cJSON *coord = pos->child;
        for (size_t i = 0; coord && (i < 3); coord = coord->next, ++i)
        {
            camera->pos._layout[i] = coord->valuedouble;
        }
    }

    cJSON *angle = cJSON_GetObjectItemCaseSensitive(object, "angle");
    if (!angle || !cJSON_IsArray(angle))
    {
        camera->angle = mt_vec3_zero();
    }
    else
    {
        cJSON *coord = angle->child;
        for (size_t i = 0; coord && (i < 3); coord = coord->next, ++i)
        {
            camera->angle._layout[i] = coord->valuedouble;
        }
    }

    cJSON *fov = cJSON_GetObjectItemCaseSensitive(object, "fov");
    if (!fov || !cJSON_IsNumber(fov))
    {
        camera->r_fov = MT_RADIANS(90.f);
    }
    else
    {
        camera->r_fov = MT_RADIANS(fov->valuedouble);
    }

    cJSON *near = cJSON_GetObjectItemCaseSensitive(object, "near");
    if (!near || !cJSON_IsNumber(near))
    {
        camera->near = 0.01f;
    }
    else
    {
        camera->near = near->valuedouble;
    }

    cJSON *far = cJSON_GetObjectItemCaseSensitive(object, "far");
    if (!far || !cJSON_IsNumber(far))
    {
        camera->far = 50.f;
    }
    else
    {
        camera->far = far->valuedouble;
    }

    return camera;
}

mvec3 mt_camera_get_forward(mt_camera *camera)
{
    mquat yaw = mt_quat_axis_angle(mt_vec3(0.f, 1.f, 0.f), camera->angle.y);

    mvec3 forward = mt_vec3_normalize(mt_quat_rotate_vec3(yaw, mt_vec3(0.f, 0.f, -1.f)));
    mvec3 local_right = mt_vec3_cross(forward, mt_vec3(0.f, 1.f, 0.f));

    mquat pitch = mt_quat_axis_angle(local_right, camera->angle.x);
    return mt_vec3_normalize(mt_quat_rotate_vec3(pitch, forward));
}

mmat4 mt_camera_get_view_projection(mt_camera *camera, float aspect)
{
    return mt_mat4_mul(mt_project_reverse(camera->r_fov, aspect, camera->near, camera->far),
                       mt_view(camera->pos, mt_camera_get_forward(camera)));
}
