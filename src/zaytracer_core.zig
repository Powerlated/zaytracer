const std = @import("std");
const main = @import("main.zig");
const assert = std.debug.assert;

const Rgb = struct {
    r: f32,
    g: f32,
    b: f32,

    fn to_argb8888(self: Rgb, a: f32) u32 {
        return (@as(u32, @as(u8, @intFromFloat(std.math.clamp(a, 0.0, 1.0) * 255.0 + 0.5))) << 24) |
            (@as(u32, @as(u8, @intFromFloat(std.math.clamp(self.r, 0.0, 1.0) * 255.0 + 0.5))) << 16) |
            (@as(u32, @as(u8, @intFromFloat(std.math.clamp(self.g, 0.0, 1.0) * 255.0 + 0.5))) << 8) |
            (@as(u32, @as(u8, @intFromFloat(std.math.clamp(self.b, 0.0, 1.0) * 255.0 + 0.5))));
    }
};

/// Linearly maps a value from a source range to a target range
fn mapRange(comptime T: type, s: T, a1: T, a2: T, b1: T, b2: T) T {
    return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

/// h: [0, 360)
/// s: [0, 1]
/// v: [0, 1]
fn hsv_to_rgb(h: f32, s: f32, v: f32) Rgb {
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
    var p: Rgb = undefined;

    if (h < 60) {
        p = .{ .r = c, .g = x, .b = 0 };
    } else if (h < 120) {
        p = .{ .r = x, .g = c, .b = 0 };
    } else if (h < 180) {
        p = .{ .r = 0, .g = c, .b = x };
    } else if (h < 240) {
        p = .{ .r = 0, .g = x, .b = c };
    } else if (h < 300) {
        p = .{ .r = x, .g = 0, .b = c };
    } else if (h < 360) {
        p = .{ .r = c, .g = 0, .b = x };
    }

    return .{ .r = p.r + m, .g = p.g + m, .b = p.b + m };
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
