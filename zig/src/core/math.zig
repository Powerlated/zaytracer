const std = @import("std");
const testing = std.testing;

pub const Vec3 = struct {
    val: @Vector(3, f32),

    pub fn zero() Vec3 {
        return .{ .val = .{ 0, 0, 0 } };
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) Vec3 {
        return .{ .val = .{ x, y, z } };
    }

    pub fn add(self: Vec3, b: Vec3) Vec3 {
        return .{ .val = self.val + b.val };
    }

    pub fn sub(self: Vec3, b: Vec3) Vec3 {
        return .{ .val = self.val - b.val };
    }

    pub fn mul_scalar(self: Vec3, b: f32) Vec3 {
        return .{ .val = self.val * @as(@Vector(3, f32), @splat(b)) };
    }

    pub fn mul_elementwise(self: Vec3, b: Vec3) Vec3 {
        return .{ .val = self.val * b.val };
    }

    pub fn div(self: Vec3, b: f32) Vec3 {
        return .{ .val = self.val / @as(@Vector(3, f32), @splat(b)) };
    }

    pub fn dot(self: Vec3, b: Vec3) f32 {
        return @reduce(.Add, self.val * b.val);
    }

    pub fn cross(self: Vec3, b_vec3: Vec3) Vec3 {
        const a = self.val;
        const b = b_vec3.val;
        const x = 0;
        const y = 1;
        const z = 2;
        return .{ .val = .{ a[y] * b[z] - a[z] * b[y], a[z] * b[x] - a[x] * b[z], a[x] * b[y] - a[y] * b[x] } };
    }

    pub fn proj(self: Vec3, v: Vec3) Vec3 {
        const u = self;
        return u.mul_scalar(v.dot(u) / u.dot(u));
    }

    pub fn normalize(self: Vec3) Vec3 {
        const magn = @sqrt(self.val[0] * self.val[0] +
            self.val[1] * self.val[1] +
            self.val[2] * self.val[2]);

        return .{ .val = self.val / @as(@Vector(3, f32), @splat(magn)) };
    }

    pub fn from_rgb_f32(r: f32, g: f32, b: f32) Rgbf32 {
        return .{ .val = .{ r, g, b } };
    }

    pub fn color_f32_to_255(color_f32: f32) u32 {
        return @as(u32, @as(u8, @intFromFloat(std.math.clamp(color_f32, 0.0, 1.0) * 255.0 + 0.5)));
    }

    pub fn to_argb8888(self: *const Rgbf32, a: f32) u32 {
        return (color_f32_to_255(a) << 24) |
            (color_f32_to_255(self.val[0]) << 16) |
            (color_f32_to_255(self.val[1]) << 8) |
            (color_f32_to_255(self.val[2]));
    }
};

pub const Mat33 = struct {
    val: @Vector(9, f32),

    pub fn det(self: Mat33) f32 {
        const v = self.val;
        const a = v[0];
        const b = v[1];
        const c = v[2];
        const d = v[3];
        const e = v[4];
        const f = v[5];
        const g = v[6];
        const h = v[7];
        const i = v[8];

        return (a * e * i + b * f * g + c * d * h) - (c * e * g + a * f * h + b * d * i);
    }

    // Calculates determinant of `self`, with column at column index (0-indexed)
    // `col_i` replaced with `col`.
    pub fn det_with_col_replaced(self: Mat33, col: Vec3, comptime col_i: u32) f32 {
        var replaced = self;

        if (!(col_i >= 0 and col_i <= 2)) {
            @compileError("col_i must be between 0 and 2 for this 3x3 matrix");
        }

        replaced.val[0 + col_i] = col.val[0];
        replaced.val[3 + col_i] = col.val[1];
        replaced.val[6 + col_i] = col.val[2];

        return replaced.det();
    }

    pub fn from_col_vecs(v0: Vec3, v1: Vec3, v2: Vec3) Mat33 {
        return .{ .val = .{
            v0.val[0], v1.val[0], v2.val[0],
            v0.val[1], v1.val[1], v2.val[1],
            v0.val[2], v1.val[2], v2.val[2],
        } };
    }

    pub fn ident() Mat33 {
        return .{ .val = .{
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        } };
    }

    pub fn mul_mat(self: Mat33, b_mat: Mat33) Mat33 {
        const a = self.val;
        const b = b_mat.val;

        var mat: Mat33 = undefined;
        inline for (0..3) |i| {
            inline for (0..3) |j| {
                mat.val[i * 3 + j] =
                    a[i * 3 + 0] * b[0 + j] +
                    a[i * 3 + 1] * b[3 + j] +
                    a[i * 3 + 2] * b[6 + j];
            }
        }

        return mat;
    }

    pub fn mul_vec(self: Mat33, b_vec: Vec3) Vec3 {
        const a = self.val;
        const b = b_vec.val;

        var vec: Vec3 = undefined;
        inline for (0..3) |j| {
            vec.val[j] =
                a[j * 3 + 0] * b[0] +
                a[j * 3 + 1] * b[1] +
                a[j * 3 + 2] * b[2];
        }

        return vec;
    }

    test "identity is identity" {
        try testing.expect(
            @reduce(.And, Vec3.from_xyz(1, 2, 3).val ==
                Mat33.ident().mul_vec(Vec3.from_xyz(1, 2, 3)).val),
        );
    }

    // Rotate about X-axis
    pub fn pitch(theta: f32) Mat33 {
        return .{ .val = .{
            1, 0,         0,
            0, @cos(theta), -@sin(theta),
            0, @sin(theta), @cos(theta),
        } };
    }

    // Rotate about Y-axis
    pub fn yaw(psi: f32) Mat33 {
        return .{ .val = .{
            @cos(psi),  0, @sin(psi),
            0,            1, 0,
            -@sin(psi), 0, @cos(psi),
        } };
    }

    // Rotate about Z-axis
    pub fn roll(phi: f32) Mat33 {
        return .{ .val = .{
            @cos(phi), -@sin(phi), 0,
            @sin(phi), @cos(phi),  0,
            0,         0,          1,
        } };
    }

    pub fn rotation(pitch_theta: f32, yaw_psi: f32, roll_phi: f32) Mat33 {
        return Mat33.yaw(yaw_psi)
            .mul_mat(Mat33.pitch(pitch_theta))
            .mul_mat(Mat33.roll(roll_phi));
    }
};

pub const Rgbf32 = Vec3;
