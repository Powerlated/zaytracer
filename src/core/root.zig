const std = @import("std");
const testing = std.testing;

const main = @import("../main.zig");

const math = @import("math.zig");
const Rgbf32 = math.Rgbf32;
const Mat33 = math.Mat33;
const Vec3 = math.Vec3;

const assert = std.debug.assert;

const MaterialShader = struct {
    color: Rgbf32,
    emission: f32,

    fn new_diffuse(r: f32, g: f32, b: f32) MaterialShader {
        return .{ .color = .{ .val = .{ r, g, b } }, .emission = 0 };
    }

    fn new_emissive(r: f32, g: f32, b: f32, emission: f32) MaterialShader {
        return .{ .color = .{ .val = .{ r, g, b } }, .emission = emission };
    }
};

const GeoTypeTag = enum(u8) {
    tri = 1,
    // sphere = 2,
};

const Geo = union(GeoTypeTag) {
    tri: Tri,
    // sphere: Sphere,
};

const TriShaderTypeTag = enum(u8) {
    perVertex = 1,
    material = 2,
};

const TriShader = union(TriShaderTypeTag) {
    perVertex: TriPerVertexShader,
    material: MaterialShader,
};

const TriPerVertexShader = struct {
    // Per-vertex color
    colors: [3]@Vector(4, f32) = @splat(@splat(0)),
};

const TriIntersection = struct {
    w: f32,
    u: f32,
    v: f32,
};

const IntersectionGeo = union(GeoTypeTag) {
    tri: TriIntersection,
    // sphere: null
};

const Intersection = struct {
    p: Vec3, // Point of intersection
    n: Vec3, // Normalized normal vector of intersection surface
    t: f32, // t value of intersection
    geo: IntersectionGeo,
};

const Tri = struct {
    // Vertices. typical notation in math is (A, B, C)
    v: [3]Vec3 = @splat(Vec3.from_xyz(0, 0, 0)),
    shader: TriShader,
    // TODO: derive normal direction from winding order?
    // n: Vec3, // n^\hat is normalized normal vector

    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    // `O`: Origin point
    // `D`: Direction vector
    fn intersect(self: *const Tri, O: Vec3, D: Vec3) ?Intersection {
        const v1 = self.v[0];
        const v2 = self.v[1];
        const v3 = self.v[2];

        // **Obtain normal vector of triangle**
        // TODO: use stored n instead of calculating on the fly
        const n = (v2.sub(v1)).cross(v3.sub(v1));

        // **Check if the ray is parallel to the triangle**
        // when ray direction vector (D) is parallel to plane (i.e. D.dot(n) == 0)
        // ray does not intersect with plane at all
        const EPSILON = 1e-6;
        if (@abs(D.dot(n)) < EPSILON) {
            return null;
        }

        // **Check if the ray-plane intersection lies outside of the triangle**
        // Find intersection point on triangle in barycentric coordinates
        //      P = w*v1 + u*v2 + v*v3;
        // P is a convex combination of the triangle's vertices, meaning that
        // the coefficients w, u, v, must sum to 1. Because of this, we can
        // replace w with 1 - u - v.
        //      P = (1 - u - v)v1 + u*v2 + v*v3
        //
        // This can be massaged into a system of linear equations:
        //      O - v1 = -t*D + u*(v2 - v1) + v*(v3 - v1);
        // Keep in mind -D, v1, v2, v3, are 3-entry vectors, so we can write
        // this system in matrix-vector multiplication form.
        //
        // [ -D_x  (v2-v1)_x  (v3-v1)_x ][ t ]   [ O_x - v1_x ]
        // [ -D_y  (v2-v1)_y  (v3-v1)_y ][ u ] = [ O_y - v1_y ]
        // [ -D_z  (v2-v1)_z  (v3-v1)_z ][ v ]   [ O_z - v1_z ]
        //
        // https://en.wikipedia.org/wiki/Cramer%27s_rule
        // Cramer's rule is an explicit formula for the solution of a system of
        // linear equations with as many equations as unknown, valid whenever
        // the system has a unique solution.
        //
        // Cramer's rule:
        // Consider a typical system of n linear equations for n unknowns,
        // represented in typical matrix multiplication form:
        //      Ax = b
        // x_i is the column vector of the solutions
        //      x_i = det(A_i) / det(A)
        // where A_i is the matrix formed by replacing the i-th column of A by
        // the column vector b

        const A = Mat33.from_col_vecs(D.mul_scalar(-1), v2.sub(v1), v3.sub(v1));
        const b = O.sub(v1);

        // t=x_1, u=x_2, v=x_3
        const t = A.det_with_col_replaced(b, 0) / A.det();
        const u = A.det_with_col_replaced(b, 1) / A.det();
        const v = A.det_with_col_replaced(b, 2) / A.det();

        // Return null if intersection of line with triangle is *behind* the direction vector
        if (t < 0) {
            return null;
        }

        // Recover w
        const w = 1 - u - v;

        // Return null if intersection of line with triangle is not inside triangle
        if (w < 0 or u < 0 or v < 0) {
            return null;
        }

        // Obtain intersection point, from the definition of barycentric coordinates
        const P = v1.mul_scalar(w).add(v2.mul_scalar(u)).add(v3.mul_scalar(v));

        return Intersection{ .p = P, .n = n.normalize(), .t = t, .geo = .{ .tri = .{ .w = w, .u = u, .v = v } } };
    }

    const TriTestHelpers = struct {
        const EPSILON: f32 = 1e-5;
        fn expect_intersect(tri: Tri, o: Vec3, d: Vec3, intersect_pos: @Vector(3, f32), t: f32) !void {
            const maybe_ti = tri.intersect(o, d);
            try testing.expect(maybe_ti != null);
            if (maybe_ti) |ti| {
                try testing.expectApproxEqAbs(t, ti.t, EPSILON);
                inline for (0..3) |i| {
                    try testing.expectApproxEqAbs(ti.p.val[i], intersect_pos[i], EPSILON);
                }
            }
        }
        fn expect_no_intersect(tri: Tri, o: Vec3, d: Vec3) !void {
            const maybe_ti = tri.intersect(o, d);
            if (maybe_ti) |ti| {
                std.debug.print("Intersection point at [{} {} {}]", .{ ti.p.val[0], ti.p.val[1], ti.p.val[2] });
            }
            try testing.expect(maybe_ti == null);
        }
    };

    test "intersection 1" {
        const ray_pos = Vec3.from_xyz(0, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(-1, -1, 1), Vec3.from_xyz(1, -1, 1), Vec3.from_xyz(0, 1, 1) } };

        try TriTestHelpers.expect_intersect(tri, ray_pos, ray_dir, .{ 0, 0, 1 }, 1);
    }

    test "intersection 2" {
        const ray_pos = Vec3.from_xyz(5, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(-10, -1, 1), Vec3.from_xyz(10, -1, 1), Vec3.from_xyz(0, 1, 1) } };

        try TriTestHelpers.expect_intersect(tri, ray_pos, ray_dir, .{ 5, 0, 1 }, 1);
    }

    test "intersection 3" {
        const ray_pos = Vec3.from_xyz(0, 0, 0);
        const ray_dir = Vec3.from_xyz(1, 0, 1).normalize();

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(-10, -100, 1), Vec3.from_xyz(100, -100, 1), Vec3.from_xyz(0, 100, 1) } };

        const tri2: Tri = .{ .v = .{ Vec3.from_xyz(-10, -100, 2), Vec3.from_xyz(100, -100, 2), Vec3.from_xyz(0, 100, 2) } };

        try TriTestHelpers.expect_intersect(tri, ray_pos, ray_dir, .{ 1, 0, 1 }, @sqrt(2));
        try TriTestHelpers.expect_intersect(tri2, ray_pos, ray_dir, .{ 2, 0, 2 }, @sqrt(2.0));
    }

    test "non-intersection 1" {
        const ray_pos = Vec3.from_xyz(0, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 1, -1);

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(-1, -1, 1), Vec3.from_xyz(1, -1, 1), Vec3.from_xyz(0, 1, 1) } };

        try TriTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
    }

    test "non-intersection 2" {
        const ray_pos = Vec3.from_xyz(5, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(-10, -1, 1), Vec3.from_xyz(1, -1, 1), Vec3.from_xyz(0, 1, 1) } };

        try TriTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
    }

    test "non-intersection 3" {
        const ray_pos = Vec3.from_xyz(1, -1, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Tri = .{ .v = .{ Vec3.from_xyz(0.5, 0.5, 1), Vec3.from_xyz(-0.5, 0.5, 1), Vec3.from_xyz(0, -0.5, 1) } };

        try TriTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
    }
};

const Sphere = struct {
    center: Vec3, // typical notation in math is (h, k, j)
    radius: f32,
};

/// Linearly maps a value from a source range to a target range
fn mapRange(comptime T: type, s: T, a1: T, a2: T, b1: T, b2: T) T {
    return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

/// h: [0, 360)
/// s: [0, 1]
/// v: [0, 1]
fn hsv_to_rgb(h: f32, s: f32, v: f32) Rgbf32 {
    assert(h >= 0 and h < 360);
    assert(s >= 0 and s <= 1);
    assert(v >= 0 and v <= 1);

    // calculate chroma
    const c = v * s;

    // calculate secondary color component
    const x = c * (1 - @abs(@mod(h / 60, 2) - 1));

    // calculate match value
    const m = v - c;

    // r', g', b'
    var p: @Vector(3, f32) = undefined;

    if (h < 60) {
        p = .{ c, x, 0 };
    } else if (h < 120) {
        p = .{ x, c, 0 };
    } else if (h < 180) {
        p = .{ 0, c, x };
    } else if (h < 240) {
        p = .{ 0, x, c };
    } else if (h < 300) {
        p = .{ x, 0, c };
    } else if (h < 360) {
        p = .{ c, 0, x };
    }

    return .{ .val = p + @as(@Vector(3, f32), @splat(m)) };
}

var geos: std.ArrayList(Geo) = undefined;
const GeosHandle = struct {
    indices: []u32,

    fn rotate(self: GeosHandle, pitch_deg: f32, yaw_deg: f32, roll_deg: f32) GeosHandle {
        const rot = Mat33.rotation(degToRad(pitch_deg), degToRad(yaw_deg), degToRad(roll_deg));

        for (self.indices) |hGeo| {
            for (&geos.items[hGeo].tri.v) |*v| {
                v.* = rot.mul_vec(v.*);
            }
        }

        return self;
    }

    fn translate(self: GeosHandle, x: f32, y: f32, z: f32) GeosHandle {
        for (self.indices) |hGeo| {
            for (&geos.items[hGeo].tri.v) |*v| {
                v.* = v.add(Vec3.from_xyz(x, y, z));
            }
        }

        return self;
    }
};

var jitter_prng: std.Random.DefaultPrng = undefined;
var dither_prng: std.Random.DefaultPrng = undefined;

fn add_trongle() !void {
    try geos.append(main.gpa, Geo{
        .tri = .{ .v = .{ Vec3.from_xyz(0.5, -0.5, 1), Vec3.from_xyz(-0.5, -0.5, 1), Vec3.from_xyz(0, 0.5, 1) }, .shader = .{ .perVertex = .{ .colors = .{ .{ 1, 0, 0, 0.5 }, .{ 0, 1, 0, 0.5 }, .{ 0, 0, 1, 0.5 } } } } },
    });
}

fn add_quad(len: f32, shader: TriShader) !GeosHandle {
    const half_len = len;

    const t1 = Tri{
        .v = .{ Vec3.from_xyz(-half_len / 2, -half_len / 2, 0), Vec3.from_xyz(-half_len / 2, half_len / 2, 0), Vec3.from_xyz(half_len / 2, half_len / 2, 0) },
        .shader = shader,
    };

    const t2 = Tri{
        .v = .{
            Vec3.from_xyz(half_len / 2, -half_len / 2, 0),
            Vec3.from_xyz(-half_len / 2, -half_len / 2, 0),
            Vec3.from_xyz(half_len / 2, half_len / 2, 0),
        },
        .shader = shader,
    };

    const g1: Geo = .{ .tri = t1 };
    const g2: Geo = .{ .tri = t2 };

    const indices = try main.gpa.alloc(u32, 2);
    indices[0] = @intCast(geos.items.len + 0);
    indices[1] = @intCast(geos.items.len + 1);

    try geos.appendSlice(main.gpa, &[_]Geo{ g1, g2 });

    return .{ .indices = indices };
}

pub fn zaytrace_init() !void {
    geos = .empty;
    // geo.append(root.gpa, Sphere{.center = } );

    jitter_prng = .init(blk: {
        // start from seed 0 always
        break :blk 0;
    });

    // start from seed 0 always
    dither_prng = .init(blk: {
        break :blk 0;
    });

    // try add_trongle();

    _ = (try add_quad(0.5, .{ .material = MaterialShader.new_emissive(1, 1, 1, 2) })).rotate(70, 0, 0).translate(0, -0.75, 1.25);

    // *CORNELL BOX MENTIONED*
    _ = (try add_quad(2, .{ .material = MaterialShader.new_diffuse(1, 0, 0) })).rotate( 90,  0,  0).translate( 1, 0, 1.25);   // left wall
    _ = (try add_quad(2, .{ .material = MaterialShader.new_diffuse(0, 1, 0) })).rotate(-90,  0,  0).translate(-1, 0, 1.25);  // right wall

    _ = (try add_quad(2, .{ .material = MaterialShader.new_diffuse(1, 1, 1) })).rotate(  0, 0,  90).translate(0,  1, 1.25);   // roof
    _ = (try add_quad(2, .{ .material = MaterialShader.new_diffuse(1, 1, 1) })).rotate(  0, 0, -90).translate(0, -1, 1.25);  // floor

    // try add_quad(Vec3.from_xyz(0, 0.999, 1), 1, 0, 0, 90, .{.material = MaterialShader.new_emissive(0, 0, 1, 1)}); // light

    _ = (try add_quad(2, .{ .material = MaterialShader.new_diffuse(1, 1, 1) })).rotate(  0, 90,  0).translate(0,  0, 2.25);   // rear wall
}

pub fn rainbow(buf: *[main.WINDOW_HEIGHT][main.WINDOW_WIDTH]u32) void {
    for (&buf.*, 0..) |*row_pixels, row| {
        // map row to normalized y coordinates
        //     (Vulkan style, [-1, -1] xy is the top left corner of the screen)
        const y = mapRange(f32, @floatFromInt(row), 0, main.WINDOW_HEIGHT, -1, 1);
        for (row_pixels, 0..) |*px, col| {
            const x = mapRange(f32, @floatFromInt(col), 0, main.WINDOW_WIDTH, -1, 1);
            _ = x;

            const hue_deg = mapRange(f32, y, -1, 1, 0, 360);
            px.* = hsv_to_rgb(hue_deg, 1, 1).to_argb8888(1.0);
        }
    }
}

// accumulator for antialiasing
var accumulator_buf: [main.WINDOW_HEIGHT][main.WINDOW_WIDTH]Rgbf32 =
    @splat(@splat(Rgbf32.from_rgb_f32(0, 0, 0)));
var accumulate_steps: u32 = 0;

pub const MAX_ACCUMULATE_STEPS = 1024;

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
fn random_jitter() struct { f32, f32 } {
    // prng.next() returns integer with uniform distribution in the entire u64 domain
    const next = jitter_prng.next();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    const x_jitter_int = next & 0xFFFFFFFF;
    const y_jitter_int = (next >> 32) & 0xFFFFFFFF;

    return .{
        @as(f32, @floatFromInt(x_jitter_int)) / 0xFFFFFFFF.0 - 0.5,
        @as(f32, @floatFromInt(y_jitter_int)) / 0xFFFFFFFF.0 - 0.5,
    };
}

// 2-tuple with each entry being random value in [-1, 1]
fn random_bounce() struct { f32, f32 } {
    // prng.next() returns integer with uniform distribution in the entire u64 domain
    const next = jitter_prng.next();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    const x_jitter_int = next & 0xFFFFFFFF;
    const y_jitter_int = (next >> 32) & 0xFFFFFFFF;

    return .{
        @as(f32, @floatFromInt(x_jitter_int)) / 0x7FFFFFFF.0 - 1,
        @as(f32, @floatFromInt(y_jitter_int)) / 0x7FFFFFFF.0 - 1,
    };
}

// return is random value in [0, 1]
fn random_alpha() f32 {
    // prng.next() returns integer with uniform distribution in the entire u64 domain
    const next = jitter_prng.next();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    const int = next & 0xFFFFFFFF;

    return @as(f32, @floatFromInt(int)) / 0xFFFFFFFF.0;
}

const DITHER_DIVISOR = 256;

fn update_and_dither_output_buffer(buf: *[main.WINDOW_HEIGHT][main.WINDOW_WIDTH]u32) void {
    dither_prng.seed(0);

    for (&buf.*, 0..) |*row_pixels, row| {
        for (row_pixels, 0..) |*px, col| {
            // prng.next() returns integer with uniform distribution in the entire u64 domain
            const next = dither_prng.next();

            // get bottom half to get a 32-bit var where each bit is I.I.D.
            const int = next & 0xFFFFFFFF;
            const uniform_rect_dist =
                (@as(f32, @floatFromInt(int)) / 0xFFFFFFFF.0 - 0.5) * (1 / DITHER_DIVISOR);

            var dithered = accumulator_buf[row][col].add(.{ .val = .{
                uniform_rect_dist,
                uniform_rect_dist,
                uniform_rect_dist,
            } });

            px.* = dithered.to_argb8888(1);
        }
    }
}

pub fn zaytrace_image(buf: *[main.WINDOW_HEIGHT][main.WINDOW_WIDTH]u32) u32 {
    if (accumulate_steps < MAX_ACCUMULATE_STEPS) {
        // **First, trace one ray per pixel, and then update the accumulator
        // buffer with the cumulative average.**
        for (&buf.*, 0..) |*row_pixels, row| {
            // map row to normalized y coordinates
            //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
            //      and z points into the screen, away from the viewer.)

            for (row_pixels, 0..) |_, col| {
                const jitter = random_jitter();
                const row_jittered = @as(f32, @floatFromInt(row)) + jitter.@"0";
                const col_jittered = @as(f32, @floatFromInt(col)) + jitter.@"1";

                const norm_y = mapRange(f32, row_jittered, 0, main.WINDOW_HEIGHT, -1, 1);
                const norm_x = mapRange(f32, col_jittered, 0, main.WINDOW_WIDTH, -1, 1);

                // perform antialiasing accumulate
                // https://en.wikipedia.org/wiki/Moving_average#Cumulative_average
                const acc_px = &accumulator_buf[row][col];
                const new_px = zaytrace_image_inner(norm_x, norm_y);
                acc_px.* = new_px.add(acc_px.mul_scalar(@floatFromInt(accumulate_steps)))
                    .div(@floatFromInt(accumulate_steps + 1));
            }
        }

        // **Then, update the output buffer while dithering it.**
        update_and_dither_output_buffer(buf);

        accumulate_steps += 1;
    }

    return accumulate_steps;
}

fn degToRad(deg: f32) f32 {
    return deg * std.math.pi / 180;
}

// `norm_x` and `norm_y` are Vulkan-style normalized
// device coordinates, where the top left of the screen
// is (-1, -1) and the bottom right is (1, 1).
fn zaytrace_image_inner(norm_x: f32, flipped_norm_y: f32) Rgbf32 {
    // Flip y to make y up
    const norm_y = flipped_norm_y * -1;

    // **Perspective Projection**
    const FOV_HORIZONTAL_RAD: f32 = degToRad(35);

    // Given a projection plane 1 unit ahead of the camera,
    // how far away from the center of the plane are the
    // left/right (x_outer_bound) and top/bottom (y_outer_bound) edges?
    const x_outer_bound = @tan(FOV_HORIZONTAL_RAD);
    const y_outer_bound = x_outer_bound * (@as(f32, main.WINDOW_HEIGHT) / main.WINDOW_WIDTH);

    const x_dir_unnormalized = norm_x * x_outer_bound;
    const y_dir_unnormalized = norm_y * y_outer_bound;

    const ray_pos = Vec3.from_xyz(0, 0, -2);
    const ray_dir = Vec3.from_xyz(x_dir_unnormalized, y_dir_unnormalized, 1).normalize();

    return zaytrace_ray(null, ray_pos, ray_dir, 0);
}

// **Inner raytracing loop** :D
const MAX_BOUNCES = 4;
fn zaytrace_ray(origin_geo: ?*const Geo, ray_pos: Vec3, ray_dir: Vec3, bounces: u8) Rgbf32 {
    if (bounces > MAX_BOUNCES) return Rgbf32.from_rgb_f32(0, 0, 0);

    var closest_inter_t: f32 = 1000.0;
    var maybe_closest_inter: ?Intersection = null;
    var maybe_closest_inter_geo: ?*Geo = null;

    for (geos.items) |*geo| {
        if (geo == origin_geo) continue;

        switch (geo.*) {
            .tri => |tri| {
                if (tri.intersect(ray_pos, ray_dir)) |inter| {
                    if (inter.t < closest_inter_t) {
                        // TODO: alpha only applies to direct rays so far. add separate indirect/direct bounce settings
                        if (bounces == 0) {
                            switch (tri.shader) {
                                .perVertex => |perVertex| {
                                    const w = inter.geo.tri.w;
                                    const v = inter.geo.tri.v;
                                    const u = inter.geo.tri.u;

                                    const colors = perVertex.colors;
                                    const a = w * colors[0][3] + v * colors[1][3] + u * colors[2][3];

                                    if (a > random_alpha()) {
                                        continue;
                                    }
                                },
                                else => {},
                            }
                        }

                        closest_inter_t = inter.t;
                        maybe_closest_inter = inter;
                        maybe_closest_inter_geo = geo;
                    }
                }
            },
        }
    }

    if (maybe_closest_inter) |closest_inter| {
        const tri_inter = closest_inter.geo.tri;
        if (maybe_closest_inter_geo) |geo| {
            switch (geo.*) {
                .tri => |tri| {
                    switch (tri.shader) {
                        .perVertex => |perVertex| {
                            // currently this is emissive
                            const w = tri_inter.w;
                            const v = tri_inter.v;
                            const u = tri_inter.u;

                            const colors = perVertex.colors;
                            const r = w * colors[0][0] + v * colors[1][0] + u * colors[2][0];
                            const g = w * colors[0][1] + v * colors[1][1] + u * colors[2][1];
                            const b = w * colors[0][2] + v * colors[1][2] + u * colors[2][2];

                            // make it much brighter if it's indirect bounce
                            if (bounces > 0) {
                                return Rgbf32.from_rgb_f32(r, g, b).mul_scalar(10);
                            } else {
                                return Rgbf32.from_rgb_f32(r, g, b);
                            }
                        },
                        .material => |material| {
                            if (material.emission >= 1) {
                                // Do not bounce if fully emissive
                                if (bounces > 0) {
                                    return material.color.mul_scalar(material.emission);
                                } else {
                                    // To avoid color clipping, don't multiply direct rays by more than 1
                                    return material.color;
                                }
                            } else {
                                var n = closest_inter.n;
                                if (n.dot(ray_dir) > 0) {
                                    // If normal is facing the wrong direction, flip it
                                    n = n.mul_scalar(-1);
                                }

                                // **Find orthonormal basis**
                                // Building an Orthonormal Basis, Revisited - Pixar
                                // https://jcgt.org/published/0006/01/01/
                                const x = n.val[0];
                                const y = n.val[1];
                                const z = n.val[2];
                                const sign = std.math.sign(z);
                                const a = -1.0 / (sign + z);
                                const b = x * y * a;
                                const b1 = Vec3{ .val = .{ 1 + sign * x * x * a, sign * b, -sign * x } };
                                const b2 = Vec3{ .val = .{ b, sign + y * y * a, -y } };

                                // now <n, b1, b2> is an orthonormal basis
                                // get a random 2D vector that is within the unit circle
                                // TODO there are definitely better distributions than this.
                                // TODO I just pulled this one out of my intuition
                                var random = random_bounce();
                                while (@sqrt(random.@"0" * random.@"0" + random.@"1" * random.@"1") > 1) {
                                    random = random_bounce();
                                }
                                const bounce_dir = n.add(b1.mul_scalar(random.@"0")).add(b2.mul_scalar(random.@"1")).normalize();

                                assert(bounce_dir.dot(n) > 0);
                                return material.color.mul_elementwise(zaytrace_ray(geo, closest_inter.p, bounce_dir, bounces + 1));
                                // return material.color.mul_scalar(0.5);

                                // Point-normal form vector form of plane:
                                //      n*(r-r0)=0
                                //      where n, r, and r0 are vectors
                                //      * n: normal vector
                                //      * r: any point on plane
                                //      * r0: specific point on the plane
                                // closest_inter.p is intersection point
                                // const

                                // return material.color;
                            }
                        },
                    }
                },
            }
        }
    }

    return Rgbf32.from_rgb_f32(0, 0, 0);
}

test {
    // Make sure compiler looks at tests inside structs too
    std.testing.refAllDecls(@This());
}
