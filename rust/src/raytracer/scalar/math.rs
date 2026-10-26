use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Vec3 {
    pub val: [f32; 3]
}

impl Vec3 {
    pub fn zero() -> Vec3 {
        return Vec3 { val: [0.0, 0.0, 0.0] };
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Vec3 {
        return Vec3 { val: [x, y, z] };
    }

    pub fn add(self: Vec3, b: Vec3) -> Vec3 {
        return Vec3 { val: [
            self.val[0] + b.val[0],
            self.val[1] + b.val[1],
            self.val[2] + b.val[2],
        ]};
    }

    pub fn sub(self: Vec3, b: Vec3) -> Vec3 {
        return Vec3 { val: [
            self.val[0] - b.val[0],
            self.val[1] - b.val[1],
            self.val[2] - b.val[2],
        ]};
    }

    pub fn mul_scalar(self: Vec3, b: f32) -> Vec3 {
        return Vec3 { val: [
            self.val[0] * b,
            self.val[1] * b,
            self.val[2] * b,
        ]};
    }

    pub fn mul_elementwise(self: Vec3, b: Vec3) -> Vec3 {
        return Vec3 { val: [
            self.val[0] * b.val[0],
            self.val[1] * b.val[1],
            self.val[2] * b.val[2],
        ]};
    }

    pub fn div(self: Vec3, b: f32) -> Vec3 {
        return Vec3 { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
        ]};
    }
    
    pub fn div_scalar(self: Vec3, b: f32) -> Vec3 {
        return Vec3 { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
        ]};
    }

    pub fn dot(self: Vec3, b: Vec3) -> f32 {
        return 
            self.val[0] * b.val[0] +
            self.val[1] * b.val[1] +
            self.val[2] * b.val[2];
    }

    pub fn cross(self: Vec3, b_vec3: Vec3) -> Vec3 {
        let a = self.val;
        let b = b_vec3.val;
        let x = 0;
        let y = 1;
        let z = 2;
        return Vec3 { val: [a[y] * b[z] - a[z] * b[y], a[z] * b[x] - a[x] * b[z], a[x] * b[y] - a[y] * b[x]] };
    }

    pub fn proj(self: Vec3, v: Vec3) -> Vec3 {
        let u = self;
        return u.mul_scalar(v.dot(u) / u.dot(u));
    }

    pub fn magn(self: Vec3) -> f32 {
        return f32::sqrt(self.val[0] * self.val[0] +
                         self.val[1] * self.val[1] +
                         self.val[2] * self.val[2]);
    }

    pub fn normalize(self: Vec3) -> Vec3 {
        return self.div(self.magn());
    }
}

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Bgra {
    pub val: [f32; 4]
}

impl Bgra {
    pub fn from_rgb_f32(r: f32, g: f32, b: f32) -> Bgra {
        return Bgra { val: [b, g, r, 1.0] };
    }

    pub fn zero() -> Bgra {
        return Bgra { val: [0.0, 0.0, 0.0, 0.0] };
    }

    pub fn add(self: Bgra, b: Bgra) -> Bgra {
        return Bgra { val: [
            self.val[0] + b.val[0],
            self.val[1] + b.val[1],
            self.val[2] + b.val[2],
            self.val[3] + b.val[3],
        ]};
    }

    pub fn sub(self: Bgra, b: Bgra) -> Bgra {
        return Bgra { val: [
            self.val[0] - b.val[0],
            self.val[1] - b.val[1],
            self.val[2] - b.val[2],
            self.val[3] - b.val[3],
        ]};
    }

    pub fn mul_scalar(self: Bgra, b: f32) -> Bgra {
        return Bgra { val: [
            self.val[0] * b,
            self.val[1] * b,
            self.val[2] * b,
            self.val[3] * b,
        ]};
    }

    pub fn mul_elementwise(self: Bgra, b: Bgra) -> Bgra {
        return Bgra { val: [
            self.val[0] * b.val[0],
            self.val[1] * b.val[1],
            self.val[2] * b.val[2],
            self.val[3] * b.val[3],
        ]};
    }

    pub fn div(self: Bgra, b: f32) -> Bgra {
        return Bgra { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
            self.val[3] / b,
        ]};
    }
    
    pub fn div_scalar(self: Bgra, b: f32) -> Bgra {
        return Bgra { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
            self.val[3] / b,
        ]};
    }

    pub fn color_f32_to_255(color_f32: f32) -> u32 {
        return (f32::clamp(color_f32, 0.0, 1.0) * 255.0 + 0.5) as u32;
    }

    pub fn to_argb8888(self: &Bgra) -> u32 {
        return (Self::color_f32_to_255(self.val[3]) << 24) |
            (Self::color_f32_to_255(self.val[2]) << 16) |
            (Self::color_f32_to_255(self.val[1]) << 8) |
            (Self::color_f32_to_255(self.val[0]));
    }
}

/// h: [0, 360)
/// s: [0, 1]
/// v: [0, 1]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Bgra {
    assert!(h >= 0. && h < 360.);
    assert!(s >= 0. && s <= 1.);
    assert!(v >= 0. && v <= 1.);

    // calculate chroma
    let c = v * s;

    // calculate secondary color component
    let x = c * (1. - f32::abs((h / 60.) % 2.) - 1.);

    // calculate match value
    let m = v - c;

    // r', g', b'
    let p: Bgra;

    if h < 60.  {
        p = Bgra::from_rgb_f32(c, x, 0.);
    } else if h < 120.  {
        p = Bgra::from_rgb_f32(x, c, 0.);
    } else if h < 180.  {
        p = Bgra::from_rgb_f32(0., c, x);
    } else if h < 240.  {
        p = Bgra::from_rgb_f32(0., x, c);
    } else if h < 300.  {
        p = Bgra::from_rgb_f32(x, 0., c);
    } else {
        p = Bgra::from_rgb_f32(c, 0., x);
    }

    return p.add(Bgra::from_rgb_f32(m, m, m));
}

#[derive(Clone, Copy)]
pub struct Mat33 {
    pub val: [f32; 9]
}

impl Mat33 {
    pub fn det(self: &Mat33) -> f32 {
        let v = self.val;
        let a = v[0];
        let b = v[1];
        let c = v[2];
        let d = v[3];
        let e = v[4];
        let f = v[5];
        let g = v[6];
        let h = v[7];
        let i = v[8];

        return (a * e * i + b * f * g + c * d * h) - (c * e * g + a * f * h + b * d * i);
    }

    // Calculates determinant of `self`, with column at column index (0-indexed)
    // `col_i` replaced with `col`.
    pub fn det_with_col_replaced<const COL_I: usize>(self: &Mat33, col: Vec3) -> f32 {
        let mut replaced: Mat33 = self.clone();

        const { assert!(COL_I <= 2, "col_i must be between 0 and 2 for this 3x3 matrix") };

        replaced.val[0 + COL_I] = col.val[0];
        replaced.val[3 + COL_I] = col.val[1];
        replaced.val[6 + COL_I] = col.val[2];

        return replaced.det();
    }

    pub fn from_col_vecs(v0: Vec3, v1: Vec3, v2: Vec3) -> Mat33 {
        return Mat33 { val: [
            v0.val[0], v1.val[0], v2.val[0],
            v0.val[1], v1.val[1], v2.val[1],
            v0.val[2], v1.val[2], v2.val[2],
        ]};
    }

    pub fn ident() -> Mat33 {
        return Mat33 { val: [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]};
    }

    pub fn mul_mat(self: &Mat33, b_mat: &Mat33) -> Mat33 {
        let a = self.val;
        let b = b_mat.val;

        let mut mat_val = [0.0; 9]; 
        for i in 0..3 {
            for j in 0..3 {
                mat_val[i * 3 + j] =
                    a[i * 3 + 0] * b[0 + j] +
                    a[i * 3 + 1] * b[3 + j] +
                    a[i * 3 + 2] * b[6 + j];
            }
        }

        return Mat33 { val: mat_val };
    }

    pub fn mul_vec(self: &Mat33, b_vec: Vec3) -> Vec3 {
        let a = self.val;
        let b = b_vec.val;

        let mut vec_val = [0.0; 3];
        for j in 0..3 {
            vec_val[j] =
                a[j * 3 + 0] * b[0] +
                a[j * 3 + 1] * b[1] +
                a[j * 3 + 2] * b[2];
        }

        return Vec3 { val: vec_val };
    }

    // Rotate about X-axis
    pub fn pitch(theta: f32) -> Mat33 {
        return Mat33 { val: [
            1.0, 0.0,         0.0,
            0.0, theta.cos(), -theta.sin(),
            0.0, theta.sin(), theta.cos(),
        ]};
    }

    // Rotate about Y-axis
    pub fn yaw(psi: f32) -> Mat33 {
        return Mat33 { val: [
            psi.cos(),  0.0, psi.sin(),
            0.0,        1.0, 0.0,
            -psi.sin(), 0.0, psi.cos(),
        ]};
    }

    // Rotate about Z-axis
    pub fn roll(phi: f32) -> Mat33 {
        return Mat33 { val: [
            phi.cos(), -phi.sin(), 0.0,
            phi.sin(), phi.cos(),  0.0,
            0.0,       0.0,        1.0,
        ]};
    }

    pub fn rotation(pitch_theta: f32, yaw_psi: f32, roll_phi: f32) -> Mat33 {
        return Mat33::yaw(deg_to_rad(yaw_psi))
            .mul_mat(&Mat33::pitch(deg_to_rad(pitch_theta)))
            .mul_mat(&Mat33::roll(deg_to_rad(roll_phi)));
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) -> Mat44 {
        let a = self.val;
        return Mat44 { val: [
            a[0], a[1], a[2],  x,
            a[3], a[4], a[5],  y,
            a[6], a[7], a[8],  z,
              0.,   0.,   0., 1.,
        ]};
    }
}

#[derive(Clone, Copy)]
pub struct Mat44 {
    pub val: [f32; 16]
}


#[cfg(test)]
mod tests {
    use crate::raytracer::scalar::math::{Mat33, Vec3};

    #[test]
    fn identity_is_identity() {
        let vec = Vec3::from_xyz(1.0, 2.0, 3.0);
        let transformed = Mat33::ident().mul_vec(vec);
        assert_eq!(vec.val, transformed.val);
    }
}

pub const fn deg_to_rad(deg: f32) -> f32 {
    return deg * std::f32::consts::PI / 180.;
}

/// Linearly maps a value from a source range to a target range
pub fn map_range(s: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
    return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}
