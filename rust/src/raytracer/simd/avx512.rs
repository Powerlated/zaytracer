/// My custom AVX512 intrinsics wrapper. 

use std::{arch::x86_64::*, ops::*};

pub struct F32x8 (__m512);

impl From<f32> for F32x8 {
    fn from(scalar: f32) -> F32x8 {
        unsafe { F32x8(_mm512_set1_ps(scalar)) }
    }
}

impl Add<F32x8> for F32x8 {
    type Output = F32x8;
    fn add(self, _rhs: F32x8) -> F32x8 {
        unsafe { F32x8(_mm512_add_ps(self.0, _rhs.0)) }
    }
}

impl Add<f32> for F32x8 {
    type Output = F32x8;
    fn add(self, _rhs: f32) -> F32x8 {
        unsafe { F32x8(_mm512_add_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Sub<F32x8> for F32x8 {
    type Output = F32x8;
    fn sub(self, _rhs: F32x8) -> F32x8 {
        unsafe { F32x8(_mm512_sub_ps(self.0, _rhs.0)) }
    }
}

impl Sub<f32> for F32x8 {
    type Output = F32x8;
    fn sub(self, _rhs: f32) -> F32x8 {
        unsafe { F32x8(_mm512_sub_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Mul<F32x8> for F32x8 {
    type Output = F32x8;
    fn mul(self, _rhs: F32x8) -> F32x8 {
        unsafe { F32x8(_mm512_sub_ps(self.0, _rhs.0)) }
    }
}

impl Mul<f32> for F32x8 {
    type Output = F32x8;
    fn mul(self, _rhs: f32) -> F32x8 {
        unsafe { F32x8(_mm512_mul_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Div<F32x8> for F32x8 {
    type Output = F32x8;
    fn div(self, _rhs: F32x8) -> F32x8 {
        unsafe { F32x8(_mm512_div_ps(self.0, _rhs.0)) }
    }
}

impl Div<f32> for F32x8 {
    type Output = F32x8;
    fn div(self, _rhs: f32) -> F32x8 {
        unsafe { F32x8(_mm512_div_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}