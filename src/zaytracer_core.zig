const std = @import("std");
const main = @import("main.zig");
const assert = std.debug.assert;

const Rgbf32 = struct {
    val: @Vector(3, f32),

    fn color_f32_to_255(val: f32) u32 {
        return @as(u32, @as(u8, @intFromFloat(std.math.clamp(val, 0.0, 1.0) * 255.0 + 0.5)));
    }

    fn to_argb8888(self: Rgbf32, a: f32) u32 {
        return (color_f32_to_255(a) << 24) |
            (color_f32_to_255(self.val[0]) << 16) |
            (color_f32_to_255(self.val[1]) << 8) |
            (color_f32_to_255(self.val[2]));
    }
};

const GeoTag = enum(u8) {
    triangle = 1,
    sphere = 2,
};

const Geo = union(GeoTag) {
    triangle: Triangle,
    sphere: Sphere,
};

const Triangle = struct {
    verts: [3]@Vector(3, f32), // typical notation in math is (A, B, C)
    color: [3]@Vector(3, f32),
    n: @Vector(3, f32), // nhat is normalized normal vector
};

const Sphere = struct {
    center: @Vector(3, f32), // typical notation in math is (h, k, j)
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

pub fn zaytrace_image(buf: *[main.WINDOW_HEIGHT][main.WINDOW_WIDTH]u32) void {
    for (&buf.*, 0..) |*row_pixels, row| {
        // map row to normalized y coordinates
        const y = mapRange(f32, @floatFromInt(row), 0, main.WINDOW_HEIGHT, -1, 1);
        for (row_pixels, 0..) |*px, col| {
            const x = mapRange(f32, @floatFromInt(col), 0, main.WINDOW_WIDTH, -1, 1);
            _ = x;

            const hue_deg = mapRange(f32, y, -1, 1, 0, 360);
            px.* = hsv_to_rgb(hue_deg, 1, 1).to_argb8888(1.0);
        }
    }
}
