#ifndef MASTERTOOLS_COMMON_H_
#define MASTERTOOLS_COMMON_H_

#include <math.h>

#include "mat4.h"
#include "quat.h"
#include "vec3.h"

#define MT_MAX(a, b) ((a) > (b) ? (a) : (b))

#define MT_MIN(a, b) ((a) < (b) ? (a) : (b))

#define MT_CLAMP(x, min, max) ((x) < (min) ? (min) : (x) > (max) ? (max) : (x))

static inline mmat4 mt_translate(mvec3 p)
{
    mmat4 trans = mt_mat4_identity();
    trans.m03 = p.x;
    trans.m13 = p.y;
    trans.m23 = p.z;
    return trans;
}

static inline mmat4 mt_rotation_euler_angles(float pitch, float yaw, float roll)
{
    float cosa = cosf(pitch);
    float cosb = cosf(yaw);
    float cosc = cosf(roll);

    float sina = sinf(pitch);
    float sinb = sinf(yaw);
    float sinc = sinf(roll);

    return (mmat4){
        .m0 = mt_vec4(cosc * cosb - sina * sinb * sinc, -sinc * cosa, cosc * sinb + sina * sinc * cosb, 0.f),
        .m1 = mt_vec4(sinc * cosb + cosc * sina * sinb, cosa * cosc, sinb * sinc - cosc * sina * cosb, 0.f),
        .m2 = mt_vec4(cosa * -sinb, sina, cosa * cosb, 0.f),
        .m3 = mt_vec4(0.f, 0.f, 0.f, 1.f),
    };
}

static inline mmat4 mt_rotation_axis_angle(mvec3 axis, float theta)
{
    return mt_quat_to_rotation(mt_quat_axis_angle(axis, theta));
}

static inline mmat4 mt_view(mvec3 eye, mvec3 forward)
{
    mvec3 right = mt_vec3_normalize(mt_vec3_cross(forward, mt_vec3(0.f, 1.f, 0.f)));
    mvec3 up = mt_vec3_normalize(mt_vec3_cross(right, forward));

    return (mmat4){
        .m0 = mt_vec3_to_vec4(right, -mt_vec3_dot(right, eye)),
        .m1 = mt_vec3_to_vec4(up, -mt_vec3_dot(up, eye)),
        .m2 = mt_vec3_to_vec4(mt_vec3_negate(forward), mt_vec3_dot(forward, eye)),
        .m3 = mt_vec4(0.f, 0.f, 0.f, 1.f),
    };
}

static inline mmat4 mt_unview(mvec3 eye, mvec3 forward)
{
    mvec3 right = mt_vec3_normalize(mt_vec3_cross(forward, mt_vec3(0.f, 1.f, 0.f)));
    mvec3 up = mt_vec3_normalize(mt_vec3_cross(right, forward));

    return (mmat4){
        .m0 = mt_vec4(right.x, up.x, -forward.x, eye.x),
        .m1 = mt_vec4(right.y, up.y, -forward.y, eye.y),
        .m2 = mt_vec4(right.z, up.z, -forward.z, eye.z),
        .m3 = mt_vec4(0.f, 0.f, 0.f, 1.f),
    };
}

static inline mmat4 mt_project(float r_fov, float aspect, float near, float far)
{
    float f = 1.f / tanf(0.5f * r_fov);
    float far_near_far = far / (near - far);

    return (mmat4){
        .m0 = mt_vec4(f / aspect, 0.f, 0.f, 0.f),
        .m1 = mt_vec4(0.f, f, 0.f, 0.f),
        .m2 = mt_vec4(0.f, 0.f, far_near_far, near * far_near_far),
        .m3 = mt_vec4(0.f, 0.f, -1.f, 0.f),
    };
}

static inline mmat4 mt_project_reverse(float r_fov, float aspect, float near, float far)
{
    return mt_project(r_fov, aspect, far, near);
}

#endif