use std::hint::black_box;

use raytracer::raytracer::{math::{Vec3, Vec3f32}, simd::avx512::F32x16};

#[inline(never)]
pub fn display(v: Vec3f32) {
    black_box(v);
}

pub fn main() {
    let x = black_box(Vec3f32::zero()).mul_scalar(black_box(5.));
    display(x);
}