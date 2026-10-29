use std::{arch::x86_64::*, ops::*};

use crate::raytracer::math::Bgraf32;

pub struct F32x4 (pub __m128);

impl From<&Bgraf32> for F32x4 {
    #[inline] fn from(bgraf32: &Bgraf32) -> F32x4 {
        unsafe { F32x4(_mm_loadu_ps(bgraf32.val.as_ptr()) )}
    }
}

impl From<f32> for F32x4 {
    #[inline] fn from(scalar: f32) -> F32x4 {
        unsafe { F32x4(_mm_set1_ps(scalar)) }
    }   
}

impl From<std::simd::u32x4> for F32x4 {
    #[inline] fn from(v: std::simd::u32x4) -> F32x4 {
        unsafe { F32x4(_mm_cvtepi32_ps(v.into())) }
    }
}

impl From<&[f32; 4]> for F32x4 {
    #[inline] fn from(v: &[f32; 4]) -> F32x4 {
        unsafe { F32x4(_mm_loadu_ps(v.as_ptr())) }
    }
}

impl Add<F32x4> for F32x4 {
    type Output = F32x4;
    #[inline] fn add(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_add_ps(self.0, _rhs.0)) }
    }
}

impl Add<f32> for F32x4 {
    type Output = F32x4;
    #[inline] fn add(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_add_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Sub<F32x4> for F32x4 {
    type Output = F32x4;
    #[inline] fn sub(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_sub_ps(self.0, _rhs.0)) }
    }
}

impl Sub<f32> for F32x4 {
    type Output = F32x4;
    #[inline] fn sub(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_sub_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Mul<F32x4> for F32x4 {
    type Output = F32x4;
    #[inline] fn mul(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_mul_ps(self.0, _rhs.0)) }
    }
}

impl Mul<f32> for F32x4 {
    type Output = F32x4;
    #[inline] fn mul(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_mul_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Div<F32x4> for F32x4 {
    type Output = F32x4;
    #[inline] fn div(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_div_ps(self.0, _rhs.0)) }
    }
}

impl Div<f32> for F32x4 {
    type Output = F32x4;
    #[inline] fn div(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_div_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

