use std::ops::*;

use bytemuck::{Pod, Zeroable};

use crate::raytracer::simd::avx512::F32x16;

trait Zero {
    fn zero();
}

trait One {
    fn one();
}

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(transparent)]
pub struct Vec3<T> {
    pub val: [T; 3]
}

pub type Vec3f32 = Vec3<f32>;

impl Vec3f32 {
    pub fn zero() -> Self {
        return Self { val: [0.0, 0.0, 0.0] };
    }

    pub fn splat(self) -> Vec3<F32x16> {
        return Vec3::<F32x16> { val: [
            F32x16::from(self.val[0]),
            F32x16::from(self.val[1]),
            F32x16::from(self.val[2])
        ] };
    }
}

impl Vec3<F32x16> {
    pub fn zero() -> Self {
        return Self { val: [
            F32x16::zero(), 
            F32x16::zero(), 
            F32x16::zero()
        ] };
    }
}

pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

impl Sqrt for f32 {
    type Output = f32;
    fn sqrt(self) -> Self::Output {
        self.sqrt()
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sqrt<Output = T> + Copy> Vec3<T> {
    pub fn from_xyz(x: T, y: T, z: T) -> Vec3<T> {
        return Vec3 { val: [x, y, z] };
    }

    pub fn add(self: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] + b.val[0],
            self.val[1] + b.val[1],
            self.val[2] + b.val[2],
        ]};
    }

    pub fn sub(self: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] - b.val[0],
            self.val[1] - b.val[1],
            self.val[2] - b.val[2],
        ]};
    }

    pub fn mul_scalar<U: Into<T> + Copy>(self: Vec3<T>, b: U) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] * b.into(),
            self.val[1] * b.into(),
            self.val[2] * b.into(),
        ]};
    }

    pub fn mul_elementwise(self: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] * b.val[0],
            self.val[1] * b.val[1],
            self.val[2] * b.val[2],
        ]};
    }

    pub fn div(self: Vec3<T>, b: T) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
        ]};
    }
    
    pub fn div_scalar<U: Into<T> + Copy>(self: Vec3<T>, b: U) -> Vec3<T> {
        return Vec3 { val: [
            self.val[0] / b.into(),
            self.val[1] / b.into(),
            self.val[2] / b.into(),
        ]};
    }

    pub fn dot<U: Into<Self> + Copy>(self: Vec3<T>, b: U) -> T {
        return 
            self.val[0] * b.into().val[0] +
            self.val[1] * b.into().val[1] +
            self.val[2] * b.into().val[2];
    }

    pub fn cross(self: Vec3<T>, b_vec3: Vec3<T>) -> Vec3<T> {
        let a = self.val;
        let b = b_vec3.val;
        let x = 0;
        let y = 1;
        let z = 2;
        return Vec3 { val: [a[y] * b[z] - a[z] * b[y], a[z] * b[x] - a[x] * b[z], a[x] * b[y] - a[y] * b[x]] };
    }

    pub fn proj(self: Vec3<T>, v: Vec3<T>) -> Vec3<T> {
        let u = self;
        return u.mul_scalar(v.dot(u) / u.dot(u));
    }

    pub fn magn(self: Vec3<T>) -> T {
        return (self.val[0] * self.val[0] +
                self.val[1] * self.val[1] +
                self.val[2] * self.val[2]).sqrt();
    }

    pub fn normalize(self: Vec3<T>) -> Vec3<T> {
        return self.div(self.magn());
    }
}

impl Into<Vec3<F32x16>> for Vec3f32 {
    fn into(self) -> Vec3<F32x16> {
        self.splat()
    }
}

#[derive(Clone, Copy, Default, Pod, Zeroable, Debug)]
#[repr(transparent)]
pub struct Bgra<T> {
    pub val: [T; 4]
} 

pub type Bgraf32 = Bgra<f32>;

impl Bgraf32 {
    pub fn from_rgb_f32(r: f32, g: f32, b: f32) -> Bgraf32 {
        return Bgra { val: [b, g, r, 1.0] };
    }

    pub fn zero() -> Bgraf32 {
        return Bgra { val: [0., 0., 0., 0.] };
    }

    pub fn to_argb8888(self) -> [u8; 4] {
        return [Self::color_f32_to_255(self.val[0]),
                Self::color_f32_to_255(self.val[1]),
                Self::color_f32_to_255(self.val[2]),
                Self::color_f32_to_255(self.val[3])];
    }

    pub fn splat(self) -> Bgra<F32x16> {
        return Bgra { val: [
            F32x16::splat(self.val[0]),
            F32x16::splat(self.val[1]),
            F32x16::splat(self.val[2]),
            F32x16::splat(self.val[3])
        ] };
    }
}

impl Bgra<F32x16> {
    pub fn from_rgb_f32(r: f32, g: f32, b: f32) -> Bgra<F32x16>  {
        return Bgra { val: [b.into(), g.into(), r.into(), 1.0.into()] };
    }

    pub fn zero() -> Bgra<F32x16> {
        let zero = F32x16::from(0.0);
        return Bgra { val: [zero, zero, zero, zero] };
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy> Bgra<T> {
    pub const INDEX_B: usize = 0;
    pub const INDEX_G: usize = 1;
    pub const INDEX_R: usize = 2;
    pub const INDEX_A: usize = 3;

    pub fn add(self: Bgra<T>, b: Bgra<T>) -> Bgra<T> {
        return Bgra { val: [
            self.val[0] + b.val[0],
            self.val[1] + b.val[1],
            self.val[2] + b.val[2],
            self.val[3] + b.val[3],
        ]};
    }

    pub fn sub(self: Bgra<T>, b: Bgra<T>) -> Bgra<T> {
        return Bgra { val: [
            self.val[0] - b.val[0],
            self.val[1] - b.val[1],
            self.val[2] - b.val[2],
            self.val[3] - b.val[3],
        ]};
    }

    pub fn mul_scalar(self: Bgra<T>, b: T) -> Bgra<T> {
        return Bgra { val: [
            self.val[0] * b,
            self.val[1] * b,
            self.val[2] * b,
            self.val[3] * b,
        ]};
    }

    pub fn mul_elementwise(self: Bgra<T>, b: Bgra<T>) -> Bgra<T> {
        return Bgra { val: [
            self.val[0] * b.val[0],
            self.val[1] * b.val[1],
            self.val[2] * b.val[2],
            self.val[3] * b.val[3],
        ]};
    }

    pub fn div_scalar(self: Bgra<T>, b: T) -> Bgra<T> {
        return Bgra { val: [
            self.val[0] / b,
            self.val[1] / b,
            self.val[2] / b,
            self.val[3] / b,
        ]};
    }    

    pub fn color_f32_to_255(color_f32: f32) -> u8 {
        return (f32::clamp(color_f32, 0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// h: [0, 360)
/// s: [0, 1]
/// v: [0, 1]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Bgraf32 {
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
    let p: Bgraf32;

    if h < 60.  {
        p = Bgraf32::from_rgb_f32(c, x, 0.);
    } else if h < 120.  {
        p = Bgraf32::from_rgb_f32(x, c, 0.);
    } else if h < 180.  {
        p = Bgraf32::from_rgb_f32(0., c, x);
    } else if h < 240.  {
        p = Bgraf32::from_rgb_f32(0., x, c);
    } else if h < 300.  {
        p = Bgraf32::from_rgb_f32(x, 0., c);
    } else {
        p = Bgraf32::from_rgb_f32(c, 0., x);
    }

    return p.add(Bgraf32::from_rgb_f32(m, m, m));
}

#[derive(Clone, Copy)]
pub struct Mat33<T> {
    pub val: [T; 9]
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy> Mat33<T> {
    pub fn det(&self) -> T {
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
    pub fn det_with_col_replaced<const COL_I: usize>(&self, col: Vec3<T>) -> T {
        let mut replaced: Self = self.clone();

        const { assert!(COL_I <= 2, "col_i must be between 0 and 2 for this 3x3 matrix") };

        replaced.val[0 + COL_I] = col.val[0];
        replaced.val[3 + COL_I] = col.val[1];
        replaced.val[6 + COL_I] = col.val[2];

        return replaced.det();
    }

    pub fn from_col_vecs(v0: Vec3<T>, v1: Vec3<T>, v2: Vec3<T>) -> Self {
        return Self { val: [
            v0.val[0], v1.val[0], v2.val[0],
            v0.val[1], v1.val[1], v2.val[1],
            v0.val[2], v1.val[2], v2.val[2],
        ]};
    }
}

pub type Mat33f32 = Mat33<f32>;

impl Mat33f32 {
    pub fn ident() -> Self {
        return Self { val: [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]};
    }

    pub fn mul_mat(&self, b_mat: &Self) -> Self {
        let a = self.val;
        let b = b_mat.val;

        let mut mat_val = [0.; 9]; 
        for i in 0..3 {
            for j in 0..3 {
                mat_val[i * 3 + j] =
                    a[i * 3 + 0] * b[0 + j] +
                    a[i * 3 + 1] * b[3 + j] +
                    a[i * 3 + 2] * b[6 + j];
            }
        }

        return Self { val: mat_val };
    }

    pub fn mul_vec(&self, b_vec: Vec3<f32>) -> Vec3<f32> {
        let a = self.val;
        let b = b_vec.val;

        let mut vec_val = [0.; 3];
        for j in 0..3 {
            vec_val[j] =
                a[j * 3 + 0] * b[0] +
                a[j * 3 + 1] * b[1] +
                a[j * 3 + 2] * b[2];
        }

        return Vec3 { val: vec_val };
    }

    // Rotate about X-axis
    pub fn pitch(theta: f32) -> Self {
        return Self { val: [
            1.0, 0.0,         0.0,
            0.0, theta.cos(), -theta.sin(),
            0.0, theta.sin(), theta.cos(),
        ]};
    }

    // Rotate about Y-axis
    pub fn yaw(psi: f32) -> Self {
        return Self { val: [
            psi.cos(),  0.0, psi.sin(),
            0.0,        1.0, 0.0,
            -psi.sin(), 0.0, psi.cos(),
        ]};
    }

    // Rotate about Z-axis
    pub fn roll(phi: f32) -> Self {
        return Self { val: [
            phi.cos(), -phi.sin(), 0.0,
            phi.sin(), phi.cos(),  0.0,
            0.0,       0.0,        1.0,
        ]};
    }

    pub fn rotation(pitch_theta: f32, yaw_psi: f32, roll_phi: f32) -> Self {
        return Self::yaw(deg_to_rad(yaw_psi))
            .mul_mat(&Self::pitch(deg_to_rad(pitch_theta)))
            .mul_mat(&Self::roll(deg_to_rad(roll_phi)));
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
    use crate::raytracer::math::{Mat33f32, Vec3f32};

    #[test]
    fn identity_is_identity() {
        let vec = Vec3f32::from_xyz(1.0, 2.0, 3.0);
        let transformed = Mat33f32::ident().mul_vec(vec);
        assert_eq!(vec.val, transformed.val);
    }
}

pub const fn deg_to_rad(deg: f32) -> f32 {
    return deg * std::f32::consts::PI / 180.;
}

/// Linearly maps a value from a source range to a target range
pub fn map_range<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy, U: Into<T> + Copy>(s: T, a1: U, a2: U, b1: U, b2: U) -> T {
    return b1.into() + (s - a1.into()) * (b2.into() - b1.into()) / (a2.into() - a1.into());
}
