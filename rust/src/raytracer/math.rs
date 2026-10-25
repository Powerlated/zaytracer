#[derive(Clone, Copy)]
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

    pub fn normalize(self: Vec3) -> Vec3 {
        let magn = f32::sqrt(self.val[0] * self.val[0] +
            self.val[1] * self.val[1] +
            self.val[2] * self.val[2]);

        return self.div(magn);
    }

    pub fn color_f32_to_255(color_f32: f32) -> u32 {
        return (f32::clamp(color_f32, 0.0, 1.0) * 255.0 + 0.5) as u32;
    }

    pub fn to_argb8888(self: &Rgbf32, a: f32) -> u32 {
        return (Self::color_f32_to_255(a) << 24) |
            (Self::color_f32_to_255(self.val[0]) << 16) |
            (Self::color_f32_to_255(self.val[1]) << 8) |
            (Self::color_f32_to_255(self.val[2]));
    }
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
        return Mat33::yaw(yaw_psi)
            .mul_mat(&Mat33::pitch(pitch_theta))
            .mul_mat(&Mat33::roll(roll_phi));
    }
}

#[cfg(test)]
mod tests {
    use crate::raytracer::math::{Mat33, Vec3};

    #[test]
    fn identity_is_identity() {
        let vec = Vec3::from_xyz(1.0, 2.0, 3.0);
        let transformed = Mat33::ident().mul_vec(&vec);
        assert_eq!(vec.val, transformed.val);
    }
}

pub type Rgbf32 = Vec3;
