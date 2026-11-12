#include "engine.h"

#include <SDL3/SDL_events.h>
#include <math.h>

#include "map/map.h"
#include "rohan.h"
#include "system/text.h"
#include "system/time.h"
#include "texture.h"
#include "types.h"
#include "utility/zero_init.h"
#include "window.h"

//
#include "map/camera.h"
#include "math/common.h"
#include "math/matrix.h"
#include "math/vector.h"

static const int fps = 144;
static const int w = 1280;
static const int h = 720;

bool mt_engine_create(mt_engine *engine)
{
    // rohan_init();
    MT_ZERO_INIT(engine);

    engine->timer = mt_timer_create();
    if (!engine->timer)
    {
        goto fail;
    }

    engine->window = mt_window_create("bingus", w, h, MT_WINDOW_RESIZABLE);
    if (!engine->window)
    {
        goto fail;
    }

    engine->running = true;
    return true;

fail:
    mt_engine_free(engine);
    return false;
}

void mt_engine_free(mt_engine *engine)
{
    if (!engine)
    {
        return;
    }

    mt_map_free(&engine->map);
    mt_window_free(engine->window);
    mt_timer_free(engine->timer);

    MT_ZERO_INIT(engine);
}

#include "logging.h"
#include "map/entity.h"
#include "map/node.h"
#include "math/constants.h"

mvec3 rotate_camera(mvec3 angle, float dx, float dy)
{
    const float sensitivity = 0.002f;
    const float pitch_limit = MT_RADIANS(89.f); // +- 89 degrees

    float pitch = fmaf(-dy, sensitivity, angle.x); // angle.x - dy * sensitivity;
    float yaw = fmaf(-dx, sensitivity, angle.y);   // angle.y - dx * sensitivity;

    return mt_vec3(MT_CLAMP(pitch, -pitch_limit, pitch_limit), yaw, angle.z);
}

void move_camera(mt_camera *camera, SDL_Scancode key)
{
    const mvec3 speed = mt_vec3_all(0.1f);
    mvec3 forward = mt_camera_get_forward(camera);
    mvec3 right = mt_vec3_cross(forward, mt_vec3(0.f, 1.f, 0.f));

    switch (key)
    {
    case SDL_SCANCODE_D:
        camera->pos = mt_vec3_add(camera->pos, mt_vec3_mul(right, speed));
        return;
    case SDL_SCANCODE_A:
        camera->pos = mt_vec3_sub(camera->pos, mt_vec3_mul(right, speed));
        return;
    case SDL_SCANCODE_SPACE:
        camera->pos = mt_vec3_add(camera->pos, mt_vec3_mul(mt_vec3(0.f, 1.f, 0.f), speed));
        return;
    case SDL_SCANCODE_LSHIFT:
        camera->pos = mt_vec3_sub(camera->pos, mt_vec3_mul(mt_vec3(0.f, 1.f, 0.f), speed));
        return;
    case SDL_SCANCODE_W:
        camera->pos = mt_vec3_add(camera->pos, mt_vec3_mul(forward, speed));
        return;
    case SDL_SCANCODE_S:
        camera->pos = mt_vec3_sub(camera->pos, mt_vec3_mul(forward, speed));
        return;
    default:
        return;
    }
}

#include "analytics.h"

///

typedef struct vertex vertex;
struct vertex
{
    mvec4 pos;
    mvec3 world;
    mvec2 uv;
};

#define IN_MASK(a, b, c) (((a) << 2) | ((b) << 1) | (c))

static inline vertex clip_line_near(vertex a, vertex b)
{
    mvec4 d = mt_vec4_sub(b.pos, a.pos);
    mvec4 t = mt_vec4_all(-a.pos.z / d.z);
    return (vertex){
        .pos = mt_vec4_fma(d, t, a.pos),
        .uv = mt_vec2_fma(mt_vec2_sub(b.uv, a.uv), mt_vec4_to_vec2(t), a.uv),
        .world = mt_vec3_fma(mt_vec3_sub(b.world, a.world), mt_vec4_to_vec3(t), a.world),
    };
}

static inline vertex clip_line_left(vertex a, vertex b)
{
    mvec4 d = mt_vec4_sub(b.pos, a.pos);
    mvec4 t = mt_vec4_all((-a.pos.w - a.pos.x) / (d.x + d.w));
    return (vertex){
        .pos = mt_vec4_fma(d, t, a.pos),
        .uv = mt_vec2_fma(mt_vec2_sub(b.uv, a.uv), mt_vec4_to_vec2(t), a.uv),
        .world = mt_vec3_fma(mt_vec3_sub(b.world, a.world), mt_vec4_to_vec3(t), a.world),
    };
}

static inline vertex clip_line_right(vertex a, vertex b)
{
    mvec4 d = mt_vec4_sub(b.pos, a.pos);
    mvec4 t = mt_vec4_all((a.pos.w - a.pos.x) / (d.x - b.pos.w + a.pos.w));
    return (vertex){
        .pos = mt_vec4_fma(d, t, a.pos),
        .uv = mt_vec2_fma(mt_vec2_sub(b.uv, a.uv), mt_vec4_to_vec2(t), a.uv),
        .world = mt_vec3_fma(mt_vec3_sub(b.world, a.world), mt_vec4_to_vec3(t), a.world),
    };
}

static inline vertex clip_line_top(vertex a, vertex b)
{
    mvec4 d = mt_vec4_sub(b.pos, a.pos);
    mvec4 t = mt_vec4_all((a.pos.w - a.pos.y) / (d.y - b.pos.w + a.pos.w));
    return (vertex){
        .pos = mt_vec4_fma(d, t, a.pos),
        .uv = mt_vec2_fma(mt_vec2_sub(b.uv, a.uv), mt_vec4_to_vec2(t), a.uv),
        .world = mt_vec3_fma(mt_vec3_sub(b.world, a.world), mt_vec4_to_vec3(t), a.world),
    };
}

static inline vertex clip_line_bottom(vertex a, vertex b)
{
    mvec4 d = mt_vec4_sub(b.pos, a.pos);
    mvec4 t = mt_vec4_all((-a.pos.w - a.pos.y) / (d.y + d.w));
    return (vertex){
        .pos = mt_vec4_fma(d, t, a.pos),
        .uv = mt_vec2_fma(mt_vec2_sub(b.uv, a.uv), mt_vec4_to_vec2(t), a.uv),
        .world = mt_vec3_fma(mt_vec3_sub(b.world, a.world), mt_vec4_to_vec3(t), a.world),
    };
}

mt_texture *pixels_p;

static inline void render(mt_entity *entity, vertex a, vertex b, vertex c)
{
    float iwa = 1.f / a.pos.w;
    a.pos = mt_vec4_mul(a.pos, mt_vec4_all(iwa));
    a.uv = mt_vec2_mul(a.uv, mt_vec2_all(iwa));
    a.world = mt_vec3_mul(a.world, mt_vec3_all(iwa));

    float iwb = 1.f / b.pos.w;
    b.pos = mt_vec4_mul(b.pos, mt_vec4_all(iwb));
    b.uv = mt_vec2_mul(b.uv, mt_vec2_all(iwb));
    b.world = mt_vec3_mul(b.world, mt_vec3_all(iwb));

    float iwc = 1.f / c.pos.w;
    c.pos = mt_vec4_mul(c.pos, mt_vec4_all(iwc));
    c.uv = mt_vec2_mul(c.uv, mt_vec2_all(iwc));
    c.world = mt_vec3_mul(c.world, mt_vec3_all(iwc));

    // a.pos.x = MT_CLAMP(fmaf(a.pos.x, 0.5f, 0.5f) * w, 0, w);
    // a.pos.y = MT_CLAMP(fmaf(a.pos.y, -0.5f, 0.5f) * h, 0, h);

    // b.pos.x = MT_CLAMP(fmaf(b.pos.x, 0.5f, 0.5f) * w, 0, w);
    // b.pos.y = MT_CLAMP(fmaf(b.pos.y, -0.5f, 0.5f) * h, 0, h);

    // c.pos.x = MT_CLAMP(fmaf(c.pos.x, 0.5f, 0.5f) * w, 0, w);
    // c.pos.y = MT_CLAMP(fmaf(c.pos.y, -0.5f, 0.5f) * h, 0, h);

    a.pos.x = fmaf(fmaf(a.pos.x, 0.5f, 0.5f), w - 1.f, 0.5f);
    a.pos.y = fmaf(fmaf(a.pos.y, -0.5f, 0.5f), h - 1.f, 0.5f);

    b.pos.x = fmaf(fmaf(b.pos.x, 0.5f, 0.5f), w - 1.f, 0.5f);
    b.pos.y = fmaf(fmaf(b.pos.y, -0.5f, 0.5f), h - 1.f, 0.5f);

    c.pos.x = fmaf(fmaf(c.pos.x, 0.5f, 0.5f), w - 1.f, 0.5f);
    c.pos.y = fmaf(fmaf(c.pos.y, -0.5f, 0.5f), h - 1.f, 0.5f);

    float v0[] = {a.pos.x, a.pos.y, iwa, a.uv.u, a.uv.v, a.world.x, a.world.y, a.world.z};
    float v1[] = {b.pos.x, b.pos.y, iwb, b.uv.u, b.uv.v, b.world.x, b.world.y, b.world.z};
    float v2[] = {c.pos.x, c.pos.y, iwc, c.uv.u, c.uv.v, c.world.x, c.world.y, c.world.z};
    entity->desc.render(&entity->desc, entity->shader_instance, w * sizeof(int), sizeof(int), v0, v1, v2);

    // render_line(pixels_p, a.pos.x, a.pos.y, b.pos.x, b.pos.y, 0xffcccccc);
    // render_line(pixels_p, b.pos.x, b.pos.y, c.pos.x, c.pos.y, 0xffcccccc);
    // render_line(pixels_p, c.pos.x, c.pos.y, a.pos.x, a.pos.y, 0xffcccccc);
}

static inline void clip_top(mt_entity *entity, vertex a, vertex b, vertex c)
{
    vertex d;

    switch (IN_MASK(a.pos.y <= a.pos.w, b.pos.y <= b.pos.w, c.pos.y <= c.pos.w))
    {
    case IN_MASK(0, 0, 0): // all out
        return;

    case IN_MASK(1, 1, 1): // all in
        render(entity, a, b, c);
        return;

    case IN_MASK(1, 0, 0): // a in
        render(entity, a, clip_line_top(a, b), clip_line_top(a, c));
        return;

    case IN_MASK(0, 1, 0): // b in
        render(entity, clip_line_top(a, b), b, clip_line_top(b, c));
        return;

    case IN_MASK(0, 0, 1): // c in
        render(entity, clip_line_top(a, c), clip_line_top(b, c), c);
        return;

    case IN_MASK(1, 1, 0): // c out
        d = clip_line_top(c, a);
        render(entity, d, a, b);
        render(entity, d, b, clip_line_top(c, b));
        return;

    case IN_MASK(1, 0, 1): // b out
        d = clip_line_top(b, c);
        render(entity, d, c, a);
        render(entity, d, a, clip_line_top(b, a));
        return;

    case IN_MASK(0, 1, 1): // a out
        d = clip_line_top(a, b);
        render(entity, d, b, c);
        render(entity, d, c, clip_line_top(a, c));
        return;
    }
}

static inline void clip_bottom(mt_entity *entity, vertex a, vertex b, vertex c)
{
    vertex d;

    switch (IN_MASK(a.pos.y >= -a.pos.w, b.pos.y >= -b.pos.w, c.pos.y >= -c.pos.w))
    {
    case IN_MASK(0, 0, 0): // all out
        return;

    case IN_MASK(1, 1, 1): // all in
        clip_top(entity, a, b, c);
        return;

    case IN_MASK(1, 0, 0): // a in
        clip_top(entity, a, clip_line_bottom(a, b), clip_line_bottom(a, c));
        return;

    case IN_MASK(0, 1, 0): // b in
        clip_top(entity, clip_line_bottom(a, b), b, clip_line_bottom(b, c));
        return;

    case IN_MASK(0, 0, 1): // c in
        clip_top(entity, clip_line_bottom(a, c), clip_line_bottom(b, c), c);
        return;

    case IN_MASK(1, 1, 0): // c out
        d = clip_line_bottom(c, a);
        clip_top(entity, d, a, b);
        clip_top(entity, d, b, clip_line_bottom(c, b));
        return;

    case IN_MASK(1, 0, 1): // b out
        d = clip_line_bottom(b, c);
        clip_top(entity, d, c, a);
        clip_top(entity, d, a, clip_line_bottom(b, a));
        return;

    case IN_MASK(0, 1, 1): // a out
        d = clip_line_bottom(a, b);
        clip_top(entity, d, b, c);
        clip_top(entity, d, c, clip_line_bottom(a, c));
        return;
    }
}

static inline void clip_right(mt_entity *entity, vertex a, vertex b, vertex c)
{
    vertex d;

    switch (IN_MASK(a.pos.x <= a.pos.w, b.pos.x <= b.pos.w, c.pos.x <= c.pos.w))
    {
    case IN_MASK(0, 0, 0): // all out
        return;

    case IN_MASK(1, 1, 1): // all in
        clip_bottom(entity, a, b, c);
        return;

    case IN_MASK(1, 0, 0): // a in
        clip_bottom(entity, a, clip_line_right(a, b), clip_line_right(a, c));
        return;

    case IN_MASK(0, 1, 0): // b in
        clip_bottom(entity, clip_line_right(a, b), b, clip_line_right(b, c));
        return;

    case IN_MASK(0, 0, 1): // c in
        clip_bottom(entity, clip_line_right(a, c), clip_line_right(b, c), c);
        return;

    case IN_MASK(1, 1, 0): // c out
        d = clip_line_right(c, a);
        clip_bottom(entity, d, a, b);
        clip_bottom(entity, d, b, clip_line_right(c, b));
        return;

    case IN_MASK(1, 0, 1): // b out
        d = clip_line_right(b, c);
        clip_bottom(entity, d, c, a);
        clip_bottom(entity, d, a, clip_line_right(b, a));
        return;

    case IN_MASK(0, 1, 1): // a out
        d = clip_line_right(a, b);
        clip_bottom(entity, d, b, c);
        clip_bottom(entity, d, c, clip_line_right(a, c));
        return;
    }
}

static inline void clip_left(mt_entity *entity, vertex a, vertex b, vertex c)
{
    vertex d;

    switch (IN_MASK(a.pos.x >= -a.pos.w, b.pos.x >= -b.pos.w, c.pos.x >= -c.pos.w))
    {
    case IN_MASK(0, 0, 0): // all out
        return;

    case IN_MASK(1, 1, 1): // all in
        clip_right(entity, a, b, c);
        return;

    case IN_MASK(1, 0, 0): // a in
        clip_right(entity, a, clip_line_left(a, b), clip_line_left(a, c));
        return;

    case IN_MASK(0, 1, 0): // b in
        clip_right(entity, clip_line_left(a, b), b, clip_line_left(b, c));
        return;

    case IN_MASK(0, 0, 1): // c in
        clip_right(entity, clip_line_left(a, c), clip_line_left(b, c), c);
        return;

    case IN_MASK(1, 1, 0): // c out
        d = clip_line_left(c, a);
        clip_right(entity, d, a, b);
        clip_right(entity, d, b, clip_line_left(c, b));
        return;

    case IN_MASK(1, 0, 1): // b out
        d = clip_line_left(b, c);
        clip_right(entity, d, c, a);
        clip_right(entity, d, a, clip_line_left(b, a));
        return;

    case IN_MASK(0, 1, 1): // a out
        d = clip_line_left(a, b);
        clip_right(entity, d, b, c);
        clip_right(entity, d, c, clip_line_left(a, c));
        return;
    }
}

static inline void clip_and_render(mt_entity *entity, vertex a, vertex b, vertex c)
{
    vertex d;

    switch (IN_MASK(a.pos.z >= 0.f, b.pos.z >= 0.f, c.pos.z >= 0.f))
    {
    case IN_MASK(0, 0, 0): // all out
        return;

    case IN_MASK(1, 1, 1): // all in
        clip_left(entity, a, b, c);
        return;

    case IN_MASK(1, 0, 0): // a in
        clip_left(entity, a, clip_line_near(a, b), clip_line_near(a, c));
        return;

    case IN_MASK(0, 1, 0): // b in
        clip_left(entity, clip_line_near(a, b), b, clip_line_near(b, c));
        return;

    case IN_MASK(0, 0, 1): // c in
        clip_left(entity, clip_line_near(a, c), clip_line_near(b, c), c);
        return;

    case IN_MASK(1, 1, 0): // c out
        d = clip_line_near(c, a);
        clip_left(entity, d, a, b);
        clip_left(entity, d, b, clip_line_near(c, b));
        return;

    case IN_MASK(1, 0, 1): // b out
        d = clip_line_near(b, c);
        clip_left(entity, d, c, a);
        clip_left(entity, d, a, clip_line_near(b, a));
        return;

    case IN_MASK(0, 1, 1): // a out
        d = clip_line_near(a, b);
        clip_left(entity, d, b, c);
        clip_left(entity, d, c, clip_line_near(a, c));
        return;
    }
}

bool mt_engine_run(mt_engine *engine)
{
    mt_frame_analytics analytics = mt_analytics_create(engine->timer);
    float aspect = (float)w / (float)h;

    mt_texture pixels = mt_texture_create(w, h, MT_TEXTURE_RGBA8888);
    pixels_p = &pixels;

    // mt_texture *depth = mt_texture_create(w, h, MT_TEXTURE_RGBA8888);
    float *depth = _mm_malloc(w * h * sizeof *depth, 32);

    mt_texture *cubemap = (mt_texture[6]){};
    mt_texture_load_cubemap(mt_string_refer_raw("resource/cubemap_2/"), cubemap);

    float camera_radius = 1.f;

    for (; engine->running; engine->frames_elapsed += 1)
    {
        mt_timer_reset(engine->timer);

        mt_entity *entity;
        mt_camera *active_camera = mt_array_ptr(&engine->map.cameras, mt_camera, engine->map.active_camera_id);

        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_EVENT_QUIT)
            {
                engine->running = false;
                break;
            }

            else if (e.type == SDL_EVENT_KEY_DOWN)
            {
                if (e.key.scancode == SDL_SCANCODE_Q)
                {
                    engine->running = false;
                    break;
                }
                else
                {
                    if (active_camera)
                    {
                        move_camera(active_camera, e.key.scancode);
                        // float speed = 0.25f;

                        // switch (e.key.scancode)
                        // {
                        // case SDL_SCANCODE_A:
                        //     active_camera->angle.y -= speed;
                        //     break;
                        // case SDL_SCANCODE_D:
                        //     active_camera->angle.y += speed;
                        //     break;
                        // case SDL_SCANCODE_W:
                        //     active_camera->angle.x = fmaxf(active_camera->angle.x - speed * 0.25f,
                        //     MT_RADIANS(-80.f)); break;
                        // case SDL_SCANCODE_S:
                        //     active_camera->angle.x = fminf(active_camera->angle.x + speed * 0.25f, MT_RADIANS(80.f));
                        //     break;
                        // default:
                        //     break;
                        // }
                    }
                }
            }

            else if (e.type == SDL_EVENT_MOUSE_WHEEL)
            {
                camera_radius = fmaxf(camera_radius - e.wheel.y * 0.25f, 0.05f);
            }

            else if (e.type == SDL_EVENT_MOUSE_MOTION)
            {
                if (active_camera)
                {
                    active_camera->angle = rotate_camera(active_camera->angle, e.motion.xrel, e.motion.yrel);
                }
            }
        }

        mvec3 camera_forward = mt_camera_get_forward(active_camera);

        float s = 0.002f;
        mvec3 light_pos = mt_vec3(1.5f * cosf(engine->time_elapsed * s), 0.f, 1.5f * sinf(engine->time_elapsed * s));
        // mvec3 light_pos = mt_vec3(1.f, 0.f, 1.f);

        if (active_camera)
        {
            // active_camera->pos =
            //     mt_vec3_fms(mt_vec3_negate(camera_forward), mt_vec3_all(camera_radius), active_camera->center);

            mmat4 projection = mt_project(active_camera->r_fov, aspect, active_camera->near, active_camera->far);
            mmat4 view_projection = mt_mat4_mul(projection, mt_view(active_camera->pos, camera_forward));

            mt_array_foreach(&engine->map.nodes, mt_node, node)
            {
                mt_node_propagate_transform(node, &engine->map, mt_mat4_identity(),
                                            mt_timer_get_elapsed_since_created(engine->timer));

                if (node->kind != MT_NODE_ENTITY)
                {
                    continue;
                }

                entity = node->data;
                rohan_set_uniform(&entity->desc, entity->shader_instance, 0, &pixels);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 1, &depth);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 3, &entity->texture);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 4, &cubemap);

                __m256 lx = _mm256_set1_ps(light_pos.x);
                __m256 ly = _mm256_set1_ps(light_pos.y);
                __m256 lz = _mm256_set1_ps(light_pos.z);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 5, &lx);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 6, &ly);
                rohan_set_uniform(&entity->desc, entity->shader_instance, 7, &lz);

                mmat4 mvp = mt_mat4_mul(node->view_parented ? projection : view_projection, node->transform);

                mt_array_foreach(&entity->meshes, mt_mesh, mesh)
                {
                    uint *indices = mesh->indices.data;
                    for (size_t i = 0; i < mesh->indices.size; i += 7)
                    {
                        rohan_set_uniform(&entity->desc, entity->shader_instance, 2, &indices[i]);
                        vertex vertices[3];

                        for (size_t j = 0; j < 3; ++j)
                        {
                            size_t k = i + j * 2;
                            mvec3 xyz = mt_vec3_load((float *)mesh->vertices.data + indices[k + 1] * 3);
                            vertices[j].pos = mt_mat4_mul_vec4(mvp, mt_vec3_to_vec4(xyz, 1.f));
                            vertices[j].uv = mt_vec2_load((float *)mesh->uvs.data + indices[k + 2] * 2);
                            vertices[j].world = mt_vec3_sub(light_pos, xyz);
                        }

                        vertex a = vertices[0], b = vertices[1], c = vertices[2];

                        // float D = a.pos.x * (b.pos.y * c.pos.w - c.pos.y * b.pos.w) -
                        //           a.pos.y * (b.pos.x * c.pos.w - c.pos.x * b.pos.w) +
                        //           a.pos.w * (b.pos.x * c.pos.y - c.pos.x * b.pos.y);

                        mvec3 A = mt_vec3(a.pos.x, a.pos.y, a.pos.w);
                        mvec3 B = mt_vec3(b.pos.y, b.pos.x, b.pos.x);
                        mvec3 C = mt_vec3(c.pos.w, c.pos.w, c.pos.y);
                        mvec3 D = mt_vec3(c.pos.y, c.pos.x, c.pos.x);
                        mvec3 E = mt_vec3(b.pos.w, b.pos.w, b.pos.y);
                        mvec3 R = mt_vec3_mul(A, mt_vec3_fms(B, C, mt_vec3_mul(D, E)));

                        if (R.x - R.y + R.z > 0.f)
                        {
                            mvec3 ab = mt_vec3_normalize(mt_vec3_sub(b.world, a.world));
                            mvec3 bc = mt_vec3_normalize(mt_vec3_sub(c.world, b.world));
                            mvec3 normal = mt_vec3_normalize(mt_vec3_cross(ab, bc));

                            __m256 nx = _mm256_set1_ps(normal.x), ny = _mm256_set1_ps(normal.y),
                                   nz = _mm256_set1_ps(normal.z);
                            rohan_set_uniform(&entity->desc, entity->shader_instance, 8, &nx);
                            rohan_set_uniform(&entity->desc, entity->shader_instance, 9, &ny);
                            rohan_set_uniform(&entity->desc, entity->shader_instance, 10, &nz);

                            clip_and_render(entity, a, b, c);
                        }
                    }
                }
            }

            mvec4 proj_light = mt_mat4_mul_vec4(view_projection, mt_vec3_to_vec4(light_pos, 1.f));
            if (proj_light.w > 0.f)
            {
                mt_string_view text = mt_string_refer_raw("LIGHT");
                int size = 10;
                int width = text.len * size;

                float lx = fmaf(fmaf(proj_light.x / proj_light.w, 0.5f, 0.5f), w - 1.f, 0.5f) - width * 0.5f;
                float ly = fmaf(fmaf(proj_light.y / proj_light.w, -0.5f, 0.5f), h - 1.f, 0.5f);

                if (lx >= 0.f && (lx + width) < w && ly >= 0.f && (ly + size) < h)
                {
                    mt_render_text_2d_default(&pixels, text, lx, ly, size, size, 0xff00ffff);
                }
            }
        }

        // mvec3 top_left, top_right, bottom_left, bottom_right;
        // {
        //     mvec3 center =
        //         mt_vec3_add(active_camera->pos, mt_vec3_mul(camera_forward, mt_vec3_all(active_camera->near)));
        //     mvec3 half_h = mt_vec3_all(active_camera->near * tanf(active_camera->r_fov * 0.5f));
        //     mvec3 half_w = mt_vec3_mul(half_h, mt_vec3_all(aspect));

        //     mvec3 right = mt_vec3_normalize(mt_vec3_cross(camera_forward, mt_vec3(0.f, 1.f, 0.f)));
        //     mvec3 up = mt_vec3_normalize(mt_vec3_cross(camera_forward, right));

        //     mvec3 r_half_w = mt_vec3_mul(right, half_w);
        //     mvec3 u_half_h = mt_vec3_mul(up, half_h);

        //     top_left = mt_vec3_sub(mt_vec3_sub(center, r_half_w), u_half_h);
        //     top_right = mt_vec3_sub(mt_vec3_add(center, r_half_w), u_half_h);
        //     bottom_left = mt_vec3_add(mt_vec3_sub(center, r_half_w), u_half_h);
        //     bottom_right = mt_vec3_add(mt_vec3_add(center, r_half_w), u_half_h);

        //     top_left = mt_vec3_sub(top_left, active_camera->pos);
        //     top_right = mt_vec3_sub(top_right, active_camera->pos);
        //     bottom_left = mt_vec3_sub(bottom_left, active_camera->pos);
        //     bottom_right = mt_vec3_sub(bottom_right, active_camera->pos);
        // }

        // float v0[] = {0.f, 0.f, top_left.x, top_left.y, top_left.z};
        // float v1[] = {w, 0.f, top_right.x, top_right.y, top_right.z};
        // float v2[] = {w, h, bottom_right.x, bottom_right.y, bottom_right.z};
        // float v3[] = {0, h, bottom_left.x, bottom_left.y, bottom_left.z};

        // entity->desc.render(&entity->desc, entity->shader_instance, w * sizeof(int), sizeof(int), v0, v1, v2);
        // entity->desc.render(&entity->desc, entity->shader_instance, w * sizeof(int), sizeof(int), v0, v2, v3);

        if (true)
        {
            int written_total = 0;
            int written_now = 0;

            char text_buf[64];
            written_now = sprintf(text_buf, "ms.a: %.3f ", mt_analytics_get_average_frametime(&analytics));
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), 10, 10, 10, 10, 0xff0000ff);

            written_total += written_now + 1;
            written_now = sprintf(text_buf, "ms.r: %.3f ", engine->delta_time);
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), written_total * 10, 10, 10, 10,
                                      0xff0000ff);

            written_total += written_now;
            written_now = sprintf(text_buf, "fps: %zu ", (size_t)(MT_MILLIS_PER_SEC / engine->delta_time));
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), written_total * 10, 10, 10, 10,
                                      0xff00ccff);

            written_total += written_now;
            written_now = sprintf(text_buf, "frame: %zu", engine->frames_elapsed);
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), written_total * 10, 10, 10, 10,
                                      0xff00ff00);

            written_total = 0;
            written_now = sprintf(text_buf, "t: %dx%d@%dHz ", w, h, fps);
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), 10, 25, 10, 10, 0xffcccccc);

            written_total += written_now + 1;
            written_now = sprintf(text_buf, "w: %dx%d", 1440, 900);
            mt_render_text_2d_default(&pixels, mt_string_refer_raw(text_buf), 10 * written_total, 25, 10, 10,
                                      0xffcccccc);

            mt_analytics_render_frame_graph(&analytics, &pixels, 10, 50, 200, 50);
        }

        mt_window_render(engine->window, &pixels);

        mt_texture_clear(&pixels, 0xff888888);

        //  mt_texture_clear(depth, UINT32_MAX);
        // memset(depth, 0xff, w * h * sizeof(float));
        for (size_t i = 0; i < (size_t)(w * h); ++i)
        {
            depth[i] = 0.f;
        }

        // double elapsed = mt_timer_get_elapsed(engine->timer);
        // double frame_surplus = MT_MILLIS_PER_SEC / (double)fps - elapsed;
        // if (frame_surplus > 0.0)
        // {
        //     mt_sleep((size_t)frame_surplus);
        // }

        engine->delta_time = mt_timer_get_elapsed(engine->timer);
        engine->time_elapsed += engine->delta_time;
    }

    // LINFO("%f", engine->time_elapsed / engine->frames_elapsed);

    mt_texture_free(&pixels);
    // mt_texture_free(depth);

    return true;
}
