const std = @import("std");
const testing = std.testing;
const main = @import("main.zig");
const assert = std.debug.assert;

const Vec3 = struct {
    val: @Vector(3, f32),

    fn from_xyz(x: f32, y: f32, z: f32) Vec3 {
        return .{ .val = .{x, y, z}};
    }

    fn add(self: Vec3, b: Vec3) Vec3 {
        return .{ .val = self.val + b.val };
    }

    fn sub(self: Vec3, b: Vec3) Vec3 {
        return .{ .val = self.val - b.val };
    }

    fn mul(self: Vec3, b: f32) Vec3 {
        return .{ .val = self.val * @as(@Vector(3, f32), @splat(b)) };
    }

    fn div(self: Vec3, b: f32) Vec3 {
        return .{ .val = self.val / @as(@Vector(3, f32), @splat(b)) };
    }

    fn dot(self: Vec3, b: Vec3) f32 {
        return @reduce(.Add, self.val * b.val);
    }

    fn cross(self: Vec3, b_vec3: Vec3) Vec3 {
        const a = self.val;
        const b = b_vec3.val;
        const x = 0; const y = 1; const z = 2;
        return .{ .val = .{
            a[y]*b[z]-a[z]*b[y],
            a[z]*b[x]-a[x]*b[z],
            a[x]*b[y]-a[y]*b[x]
        } };
    }

    fn normalize(self: Vec3) Vec3 {
        const magn = @sqrt(self.val[0]*self.val[0] +
                           self.val[1]*self.val[1] +
                           self.val[2]*self.val[2]);

        return .{ .val = self.val / @as(@Vector(3, f32), @splat(magn)) };
    }

    fn from_rgb_f32(r: f32, g: f32, b: f32) Rgbf32 {
        return .{ .val = .{r, g, b} };
    }

    fn color_f32_to_255(color_f32: f32) u32 {
        return @as(u32, @as(u8, @intFromFloat(std.math.clamp(color_f32, 0.0, 1.0) * 255.0 + 0.5)));
    }

    fn to_argb8888(self: Rgbf32, a: f32) u32 {
        return (color_f32_to_255(a) << 24) |
            (color_f32_to_255(self.val[0]) << 16) |
            (color_f32_to_255(self.val[1]) << 8) |
            (color_f32_to_255(self.val[2]));
    }
};

const Mat33 = struct {
    val: @Vector(9, f32),
    
    fn det(self: Mat33) f32 {
        const v = self.val;
        const a = v[0]; const b = v[1]; const c = v[2];
        const d = v[3]; const e = v[4]; const f = v[5];
        const g = v[6]; const h = v[7]; const i = v[8];
        
        return (a*e*i + b*f*g + c*d*h) - (c*e*g + a*f*h + b*d*i);
    }

    // Calculates determinant of `self`, with column at column index (0-indexed)
    // `col_i` replaced with `col`.
    fn det_with_col_replaced(self: Mat33, col: Vec3, comptime col_i: u32) f32 {
        var replaced = self;

        if (!(col_i >= 0 and col_i <= 2)) 
        {
            @compileError("col_i must be between 0 and 2 for this 3x3 matrix");
        }

        replaced.val[0 + col_i] = col.val[0];
        replaced.val[3 + col_i] = col.val[1];
        replaced.val[6 + col_i] = col.val[2];

        return replaced.det();
    }

    fn from_col_vecs(v0: Vec3, v1: Vec3, v2: Vec3) Mat33 {
        return .{ 
            .val = .{
                v0.val[0], v1.val[0], v2.val[0],
                v0.val[1], v1.val[1], v2.val[1],
                v0.val[2], v1.val[2], v2.val[2],
            }
        };
    }
};

const Rgbf32 = Vec3;

const GeoTag = enum(u8) {
    triangle = 1,
    // sphere = 2,
};

const Geo = union(GeoTag) {
    triangle: Triangle,
    // sphere: Sphere,
};

const TriangleIntersection = struct {
    w: f32, u: f32, v: f32,
};

const IntersectionGeo = union(GeoTag) {
    triangle: TriangleIntersection,
    // sphere: null
};

const Intersection = struct {
    p: Vec3, // Point of intersection
    n: Vec3, // Normalized normal vector of intersection surface
    geo: IntersectionGeo
};
 
const Triangle = struct {
    // Vertices. typical notation in math is (A, B, C)
    v: [3]Vec3 = @splat(Vec3.from_xyz(0,0,0)), 
    // Per-vertex color
    colors: [3]@Vector(3, f32) = @splat(@splat(0)), 
    // TODO: derive normal direction from winding order?
    // n: Vec3, // n^\hat is normalized normal vector 

    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    // `O`: Origin point
    // `D`: Direction vector
    fn intersect(self: Triangle, O: Vec3, D: Vec3) ?Intersection {
        const v1 = self.v[0]; const v2 = self.v[1]; const v3 = self.v[2];

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

        const A = Mat33.from_col_vecs(D.mul(-1), v2.sub(v1), v3.sub(v1));
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
        const P = v1.mul(w).add(v2.mul(u)).add(v3.mul(v));

        return Intersection {
            .p = P,
            .n = n.normalize(),
            .geo = .{
                .triangle = .{
                    .w=w, .u=u, .v=v
                }
            }
        };
    }

    const TriangleTestHelpers = struct {
        const EPSILON: f32 = 1e-6;
        fn expect_intersect(tri: Triangle, o: Vec3, d: Vec3, intersect_pos: @Vector(3, f32)) !void {
            const maybe_ti = tri.intersect(o, d);
            try testing.expect(maybe_ti != null);
            if (maybe_ti) |ti| {
                inline for (0..3) |i| { 
                    try testing.expectApproxEqAbs(
                        ti.p.val[i],
                        intersect_pos[i],
                        EPSILON);
                }
            }
        }
        fn expect_no_intersect(tri: Triangle, o: Vec3, d: Vec3) !void {
            const maybe_ti = tri.intersect(o, d);
            if (maybe_ti) |ti| {
                std.debug.print("Intersection point at [{} {} {}]", .{ti.p.val[0], ti.p.val[1], ti.p.val[2]});
            }
            try testing.expect(maybe_ti == null);
        }
    };

    test "intersection 1" {
        const ray_pos = Vec3.from_xyz(0, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Triangle = .{
            .v = .{
                Vec3.from_xyz(-1, -1,  1),
                Vec3.from_xyz( 1, -1,  1),
                Vec3.from_xyz( 0,  1,  1)
            }
        };

        try TriangleTestHelpers.expect_intersect(tri, ray_pos, ray_dir, .{0, 0, 1});
    }

    test "intersection 2" {
        const ray_pos = Vec3.from_xyz(5, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Triangle = .{
            .v = .{
                Vec3.from_xyz(-10, -1,  1),
                Vec3.from_xyz( 10, -1,  1),
                Vec3.from_xyz( 0,  1,  1)
            }
        };

        try TriangleTestHelpers.expect_intersect(tri, ray_pos, ray_dir, .{5, 0, 1});
    }

    test "non-intersection 1" {
        const ray_pos = Vec3.from_xyz(0, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 1, -1);

        const tri: Triangle = .{
            .v = .{
                Vec3.from_xyz(-1, -1,  1),
                Vec3.from_xyz( 1, -1,  1),
                Vec3.from_xyz( 0,  1,  1)
            }
        };

        try TriangleTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
    }

    test "non-intersection 2" {
        const ray_pos = Vec3.from_xyz(5, 0, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Triangle = .{
            .v = .{
                Vec3.from_xyz(-10, -1,  1),
                Vec3.from_xyz( 1, -1,  1),
                Vec3.from_xyz( 0,  1,  1)
            }
        };

        try TriangleTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
    }

    test "non-intersection 3" {
        const ray_pos = Vec3.from_xyz(1, -1, 0);
        const ray_dir = Vec3.from_xyz(0, 0, 1);

        const tri: Triangle = .{
            .v = .{
                Vec3.from_xyz( 0.5, 0.5,  1),
                Vec3.from_xyz(-0.5, 0.5,  1),
                Vec3.from_xyz( 0,  -0.5,  1)
            }
        };

        try TriangleTestHelpers.expect_no_intersect(tri, ray_pos, ray_dir);
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

var geo: std.ArrayList(Geo) = undefined;
var prng: std.Random.DefaultPrng = undefined;

pub fn zaytrace_init() !void {
    geo = .empty;
    // geo.append(root.gpa, Sphere{.center = } );

    prng = .init(blk: {
        // start from seed 0 always
        break :blk 0;
    });
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

pub const MAX_ACCUMULATE_STEPS = 128;

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
fn random_jitter() struct {f32, f32} {
    // prng.next() returns integer with uniform distribution in the entire u64 domain
    const next = prng.next();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    const x_jitter_int = next & 0xFFFFFFFF;
    const y_jitter_int = (next >> 32) & 0xFFFFFFFF;

    return .{
        @as(f32, @floatFromInt(x_jitter_int)) / 0x100000000 - 0.5,     
        @as(f32, @floatFromInt(y_jitter_int)) / 0x100000000 - 0.5,     
    };
}

pub fn zaytrace_image(buf: *[main.WINDOW_HEIGHT][main.WINDOW_WIDTH]u32) u32 {
    if (accumulate_steps < MAX_ACCUMULATE_STEPS) {
        for (&buf.*, 0..) |*row_pixels, row| {
            // map row to normalized y coordinates
            //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
            //      and z points into the screen, away from the viewer.)

            for (row_pixels, 0..) |*px, col| {
                const jitter = random_jitter();
                const row_jittered = @as(f32, @floatFromInt(row)) + jitter.@"0";
                const col_jittered = @as(f32, @floatFromInt(col)) + jitter.@"1";
                
                const norm_y = mapRange(f32, row_jittered, 0, main.WINDOW_HEIGHT, -1, 1);
                const norm_x = mapRange(f32, col_jittered, 0, main.WINDOW_WIDTH, -1, 1);

                // perform antialiasing accumulate
                // https://en.wikipedia.org/wiki/Moving_average#Cumulative_average
                const acc_px = &accumulator_buf[row][col];
                const new_px = zaytrace_image_inner(norm_x, norm_y);
                acc_px.* = new_px.add(
                    acc_px.mul(@floatFromInt(accumulate_steps)))
                    .div(@floatFromInt(accumulate_steps + 1));

                px.* = acc_px.*.to_argb8888(1);
            }
        }

        accumulate_steps += 1;
    }

    return accumulate_steps;
}

fn zaytrace_image_inner(norm_x: f32, norm_y: f32) Rgbf32 {
    // const SCALE_X = main.WINDOW_WIDTH;
    // const SCALE_Y = main.WINDOW_HEIGHT;
    const SCALE_X = 1;
    const SCALE_Y = 1;

    const x = SCALE_X * norm_x;
    const y = SCALE_Y * norm_y;

    const ray_pos = Vec3.from_xyz(x, y, 0);
    const ray_dir = Vec3.from_xyz(0, 0, 1);

    const tri: Triangle = .{
        .v = .{
            Vec3.from_xyz( 0.5, 0.5,  1),
            Vec3.from_xyz(-0.5, 0.5,  1),
            Vec3.from_xyz( 0,  -0.5,  1)
        },
        .colors = .{
            .{1, 0, 0},
            .{0, 1, 0},
            .{0, 0, 1}
        }
    };

    if (tri.intersect(ray_pos, ray_dir)) |inter| {
        const w = inter.geo.triangle.w;
        const v = inter.geo.triangle.v;
        const u = inter.geo.triangle.u;

        const r = w*tri.colors[0][0] + v*tri.colors[1][0] + u*tri.colors[2][0];
        const g = w*tri.colors[0][1] + v*tri.colors[1][1] + u*tri.colors[2][1];
        const b = w*tri.colors[0][2] + v*tri.colors[1][2] + u*tri.colors[2][2];
        
        return Rgbf32.from_rgb_f32(r, g, b);
    }

    return Rgbf32.from_rgb_f32(0, 0, 0);
}

test {
    // Make sure compiler looks at tests inside structs too
    std.testing.refAllDecls(@This());
}