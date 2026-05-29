#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <SDL3/SDL.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define RING_SAMPLES 112
#define TUBE_SAMPLES 72

#define FULL_TURN (3.14159 * 2.0)

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} pixel;

typedef struct {
    double x;
    double y;
    double z;
} vec3;

typedef struct {
    int width;
    int height;

    pixel *pixels;
    double *depths;
} frame_buffer;

typedef struct {
    double x;
    double y;
    double depth;
    vec3 normal;
} projected_point;

typedef struct {
    int center_x;
    int center_y;

    double scale;
    double distance;
} screen_projection;

typedef struct {
    pixel color;

    double ring_radius;
    double tube_radius;

    double x_rotation;
    double y_rotation;

    double x_spin_speed;
    double y_spin_speed;
} spinning_torus;

static const pixel BACKGROUND_COLOR = { 17, 17, 27, 255 };
static const int SCREEN_PADDING = 28;

static const double TORUS_SCREEN_SCALE = 0.28;
static const double PROJECTION_DISTANCE = 4.8;

static const double MIN_LIGHT = 0.26;

static const vec3 LIGHT_DIRECTION = { -0.45, 0.35, 0.82 };

static const pixel STAR_COLOR = { 150, 137, 134, 255 };
static const int STAR_COUNT = 180;

static const double EMPTY_DEPTH = -1.0e9;

static vec3 rotate_x(double angle, vec3 point)
{
    return (vec3) {
        point.x,
        point.y * cos(angle) - point.z * sin(angle),
        point.y * sin(angle) + point.z * cos(angle),
    };
}

static vec3 rotate_y(double angle, vec3 point)
{
    return (vec3) {
        point.x * cos(angle) + point.z * sin(angle),
        point.y,
        -point.x * sin(angle) + point.z * cos(angle),
    };
}

static vec3 unit_vector(vec3 vector)
{
    double length = sqrt(
        vector.x * vector.x +
        vector.y * vector.y +
        vector.z * vector.z
    );

    if (length == 0.0) {
        return vector;
    }

    return (vec3) {
        vector.x / length,
        vector.y / length,
        vector.z / length,
    };
}

static vec3 point_on_torus(spinning_torus torus, double ring_angle, double tube_angle)
{
    double tube_center_distance =
        torus.ring_radius + torus.tube_radius * cos(tube_angle);

    return (vec3) {
        tube_center_distance * cos(ring_angle),
        tube_center_distance * sin(ring_angle),
        torus.tube_radius * sin(tube_angle),
    };
}

static vec3 normal_on_torus(double ring_angle, double tube_angle)
{
    return (vec3) {
        cos(tube_angle) * cos(ring_angle),
        cos(tube_angle) * sin(ring_angle),
        sin(tube_angle),
    };
}

static frame_buffer *new_frame_buffer(int width, int height, pixel background)
{
    int total_pixels = width * height;
    frame_buffer *frame = malloc(sizeof(frame_buffer));

    if (frame == NULL) {
        return NULL;
    }

    frame->width = width;
    frame->height = height;

    frame->pixels = malloc(total_pixels * sizeof(*frame->pixels));
    frame->depths = malloc(total_pixels * sizeof(*frame->depths));

    if (frame->pixels == NULL || frame->depths == NULL) {
        free(frame->pixels);
        free(frame->depths);
        free(frame);
        return NULL;
    }

    for (int pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        frame->pixels[pixel_idx] = background;
        frame->depths[pixel_idx] = EMPTY_DEPTH;
    }

    return frame;
}

static void clear_frame_buffer(frame_buffer *frame, pixel background)
{
    int total_pixels = frame->width * frame->height;
    for (int pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        frame->pixels[pixel_idx] = background;
        frame->depths[pixel_idx] = EMPTY_DEPTH;
    }
}

static void destroy_frame_buffer(frame_buffer *frame)
{
    if (frame == NULL) {
        return;
    }

    free(frame->pixels);
    free(frame->depths);
    free(frame);
}

static void write_pixel(frame_buffer *frame, int x, int y, double depth, pixel color)
{
    if (x < 0 || x >= frame->width || y < 0 || y >= frame->height) {
        return;
    }
    int pixel_idx = y * frame->width + x;

    if (depth <= frame->depths[pixel_idx]) {
        return;
    }

    frame->depths[pixel_idx] = depth;
    frame->pixels[pixel_idx] = color;
}

static double edge_function(
    projected_point edge_start,
    projected_point edge_end,
    double x,
    double y
)
{
    double edge_x = edge_end.x - edge_start.x;
    double edge_y = edge_end.y - edge_start.y;
    double point_x = x - edge_start.x;
    double point_y = y - edge_start.y;

    // the magnitude is the parallelogram area made by the edge vector and the point vector
    // the sign is the side test
    return edge_x * point_y - edge_y * point_x;
}

static void draw_triangle(
    frame_buffer *frame,
    projected_point a,
    projected_point b,
    projected_point c,
    pixel color
)
{
    double triangle_signed_area = edge_function(a, b, c.x, c.y);

    if (triangle_signed_area == 0.0) {
        return;
    }

    int leftmost_pixel = (int) floor(fmin(a.x, fmin(b.x, c.x)));
    int rightmost_pixel = (int) ceil(fmax(a.x, fmax(b.x, c.x)));
    int top_pixel = (int) floor(fmin(a.y, fmin(b.y, c.y)));
    int bottom_pixel = (int) ceil(fmax(a.y, fmax(b.y, c.y)));

    if (
        rightmost_pixel < 0 || leftmost_pixel >= frame->width ||
        bottom_pixel < 0 || top_pixel >= frame->height
    )
        return;

    if (leftmost_pixel < 0)
        leftmost_pixel = 0;

    if (top_pixel < 0)
        top_pixel = 0;

    if (rightmost_pixel >= frame->width)
        rightmost_pixel = frame->width - 1;

    if (bottom_pixel >= frame->height)
        bottom_pixel = frame->height - 1;

    for (int y = top_pixel; y <= bottom_pixel; y++) {
        for (int x = leftmost_pixel; x <= rightmost_pixel; x++) {
            // test the _center_ of the pixel, not the top-left corner
            double pixel_x = x + 0.5;
            double pixel_y = y + 0.5;

            double a_weight =
                edge_function(b, c, pixel_x, pixel_y) / triangle_signed_area;
            double b_weight =
                edge_function(c, a, pixel_x, pixel_y) / triangle_signed_area;
            double c_weight =
                edge_function(a, b, pixel_x, pixel_y) / triangle_signed_area;

            // outside the triangle
            if (a_weight < 0.0 || b_weight < 0.0 || c_weight < 0.0)
                continue;

            double weighted_depth =
                a.depth * a_weight +
                b.depth * b_weight +
                c.depth * c_weight;

            vec3 weighted_normal = unit_vector((vec3) {
                a.normal.x * a_weight + b.normal.x * b_weight + c.normal.x * c_weight,
                a.normal.y * a_weight + b.normal.y * b_weight + c.normal.y * c_weight,
                a.normal.z * a_weight + b.normal.z * b_weight + c.normal.z * c_weight,
            });

            double direct_light =
                weighted_normal.x * LIGHT_DIRECTION.x +
                weighted_normal.y * LIGHT_DIRECTION.y +
                weighted_normal.z * LIGHT_DIRECTION.z;

            if (direct_light < 0.0)
                direct_light = 0.0;

            double brightness = MIN_LIGHT + (1.0 - MIN_LIGHT) * direct_light;

            pixel shaded_color = {
                (uint8_t)round(color.red * brightness),
                (uint8_t)round(color.green * brightness),
                (uint8_t)round(color.blue * brightness),
                color.alpha,
            };

            write_pixel(frame, x, y, weighted_depth, shaded_color);
        }
    }
}

static screen_projection projection_for_frame(frame_buffer *frame)
{
    int drawable_width = frame->width - SCREEN_PADDING * 2;
    int drawable_height = frame->height - SCREEN_PADDING * 2;

    int smaller_screen_side = drawable_width;
    if (drawable_height < smaller_screen_side)
        smaller_screen_side = drawable_height;

    int center_x = frame->width / 2;
    int center_y = frame->height / 2;

    return (screen_projection) {
        center_x,
        center_y,
        smaller_screen_side * TORUS_SCREEN_SCALE,
        PROJECTION_DISTANCE,
    };
}

static void draw_torus(frame_buffer *frame, screen_projection projection, spinning_torus torus)
{
    projected_point torus_points[RING_SAMPLES][TUBE_SAMPLES];
    for (int ring_idx = 0; ring_idx < RING_SAMPLES; ring_idx++) {
        double ring_angle =
            FULL_TURN * ring_idx / RING_SAMPLES;

        for (int tube_idx = 0; tube_idx < TUBE_SAMPLES; tube_idx++) {
            double tube_angle =
                FULL_TURN * tube_idx / TUBE_SAMPLES;

            vec3 point = point_on_torus(torus, ring_angle, tube_angle);
            vec3 normal = normal_on_torus(ring_angle, tube_angle);

            vec3 x_rotated_point = rotate_x(torus.x_rotation, point);
            vec3 x_rotated_normal = rotate_x(torus.x_rotation, normal);

            vec3 y_rotated_point = rotate_y(torus.y_rotation, x_rotated_point);
            vec3 y_rotated_normal = rotate_y(torus.y_rotation, x_rotated_normal);

            double perspective =
                projection.distance / (projection.distance - y_rotated_point.z);

            torus_points[ring_idx][tube_idx] = (projected_point) {
                projection.center_x + y_rotated_point.x * projection.scale * perspective,
                projection.center_y - y_rotated_point.y * projection.scale * perspective,
                y_rotated_point.z,
                y_rotated_normal,
            };
        }
    }
    for (int ring_idx = 0; ring_idx < RING_SAMPLES; ring_idx++) {
        int next_ring_idx = (ring_idx + 1) % RING_SAMPLES;

        for (int tube_idx = 0; tube_idx < TUBE_SAMPLES; tube_idx++) {
            int next_tube_idx = (tube_idx + 1) % TUBE_SAMPLES;

            projected_point top_left = torus_points[ring_idx][tube_idx];
            projected_point bottom_left = torus_points[next_ring_idx][tube_idx];
            projected_point bottom_right = torus_points[next_ring_idx][next_tube_idx];
            projected_point top_right = torus_points[ring_idx][next_tube_idx];

            draw_triangle(frame, top_left, bottom_left, bottom_right, torus.color);
            draw_triangle(frame, top_left, bottom_right, top_right, torus.color);
        }
    }
}

static bool render_frame(SDL_Renderer *renderer, SDL_Texture *texture, frame_buffer *frame)
{
    int bytes_per_image_row = frame->width * sizeof(pixel);

    if (!SDL_UpdateTexture(texture, NULL, frame->pixels, bytes_per_image_row)) {
        fprintf(stderr, "SDL_UpdateTexture failed: %s\n", SDL_GetError());
        return false;
    }

    if (!SDL_RenderTexture(renderer, texture, NULL, NULL)) {
        fprintf(stderr, "SDL_RenderTexture failed: %s\n", SDL_GetError());
        return false;
    }

    if (!SDL_RenderPresent(renderer)) {
        fprintf(stderr, "SDL_RenderPresent failed: %s\n", SDL_GetError());
        return false;
    }

    return true;
}

int main()
{
    int exit_code = 1;
    bool sdl_started = false;

    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    SDL_Texture *texture = NULL;
    frame_buffer *frame = NULL;

    spinning_torus torus = {
        .color = { 184, 166, 222, 255 },
        .ring_radius = 1.10,
        .tube_radius = 0.30,
        .x_rotation = 0.92,
        .y_rotation = 0.35,
        .x_spin_speed = 0.006,
        .y_spin_speed = 0.004,
    };

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        goto cleanup;
    }

    sdl_started = true;

    window = SDL_CreateWindow(
        "Torus",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        0
    );

    if (window == NULL) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        goto cleanup;
    }

    if (!SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED)) {
        fprintf(stderr, "SDL_SetWindowPosition failed: %s\n", SDL_GetError());
        goto cleanup;
    }

    renderer = SDL_CreateRenderer(window, NULL);

    if (renderer == NULL) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        goto cleanup;
    }

    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    );

    if (texture == NULL) {
        fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
        goto cleanup;
    }

    SDL_ShowWindow(window);

    frame = new_frame_buffer(WINDOW_WIDTH, WINDOW_HEIGHT, BACKGROUND_COLOR);

    if (frame == NULL) {
        fprintf(stderr, "Could not allocate frame buffer.\n");
        goto cleanup;
    }

    screen_projection projection = projection_for_frame(frame);
    bool is_running = true;

    while (is_running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                is_running = false;
            }
        }

        clear_frame_buffer(frame, BACKGROUND_COLOR);
        for (int star_idx = 0; star_idx < STAR_COUNT; star_idx++) {
            int x = (star_idx * star_idx) % frame->width;
            int y = (star_idx * x) % frame->height;
            int pixel_idx = y * frame->width + x;

            frame->pixels[pixel_idx] = STAR_COLOR;
        }

        torus.x_rotation += torus.x_spin_speed;
        torus.y_rotation += torus.y_spin_speed;

        draw_torus(frame, projection, torus);

        if (!render_frame(renderer, texture, frame)) {
            goto cleanup;
        }
    }

    exit_code = 0;

cleanup:
    destroy_frame_buffer(frame);

    if (texture != NULL) {
        SDL_DestroyTexture(texture);
    }

    if (renderer != NULL) {
        SDL_DestroyRenderer(renderer);
    }

    if (window != NULL) {
        SDL_DestroyWindow(window);
    }

    if (sdl_started) {
        SDL_Quit();
    }

    return exit_code;
}
