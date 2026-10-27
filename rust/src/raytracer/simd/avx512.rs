/// My custom AVX512 intrinsics wrapper. 

use std::{arch::x86_64::*, ops::*};

use crate::raytracer::math::{Bgraf32, Sqrt};

pub struct U8x16 (pub __m128i);

impl From<F32x16> for U8x16 {
    fn from(v: F32x16) -> U8x16 {
        unsafe { U8x16(_mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(v.0))) }
    }
}

impl Into<[u8; 16]> for U8x16 {
    fn into(self) -> [u8; 16] {
        let mut array = [0u8; 16];
        unsafe { _mm_storeu_epi8(array.as_mut_ptr() as *mut i8, self.0); }
        return array;
    }
}

#[derive(Clone, Copy)]
pub struct U32x16 (pub __m512i);

impl U32x16 {
    pub const WIDTH: usize = 16;

    pub fn splat(scalar: u32) -> U32x16 {
        unsafe { U32x16(_mm512_set1_epi32(scalar as i32)) }
    }

    pub fn cmp_eq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask::<_CMP_EQ_OQ>(self.0, b.into().0) }
    }

    // pub fn min<T: Into<Self>>(self, b: T) -> U32x16 {
    //     unsafe { U32x16(_mm512_min_ps(self.0, b.into().0)) }
    // }

    // pub fn max<T: Into<Self>>(self, b: T) -> U32x16 {
    //     unsafe { U32x16(_mm512_max_ps(self.0, b.into().0)) }
    // }

    // pub fn clamp<T: Into<Self>>(self, min: T, max: T) -> U32x16 {
    //     Self::max(min.into(), Self::min(self, max))
    // }
}

impl From<u32> for U32x16 {
    fn from(scalar: u32) -> U32x16 {
        Self::splat(scalar)
    }
}

impl From<std::simd::u32x16> for U32x16 {
    fn from(v: std::simd::u32x16) -> U32x16 {
        U32x16(v.into())
    }
}

impl From<__m512i> for U32x16 {
    fn from(v: __m512i) -> U32x16 {
        Self(v)
    }
}

impl Into<[u32; 16]> for U32x16 {
    fn into(self) -> [u32; 16] {
        let mut array = [0u32; 16];
        unsafe { _mm512_store_epi32(array.as_mut_ptr() as *mut i32, self.0); }
        return array;
    }
} 

impl From<&[u32; 16]> for U32x16 {
    fn from(v: &[u32; 16]) -> U32x16 {
        unsafe { U32x16(_mm512_loadu_epi32(v.as_ptr() as *const i32)) }
    }
}

impl Add<U32x16> for U32x16 {
    type Output = U32x16;
    fn add(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_add_epi32(self.0, _rhs.0)) }
    }
}

impl Add<u32> for U32x16 {
    type Output = U32x16;
    fn add(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_add_epi32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

impl Sub<U32x16> for U32x16 {
    type Output = U32x16;
    fn sub(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_sub_epi32(self.0, _rhs.0)) }
    }
}

impl Sub<u32> for U32x16 {
    type Output = U32x16;
    fn sub(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_sub_epi32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

impl Mul<U32x16> for U32x16 {
    type Output = U32x16;
    fn mul(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_mul_epu32(self.0, _rhs.0)) }
    }
}

impl Mul<u32> for U32x16 {
    type Output = U32x16;
    fn mul(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_mul_epu32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct F32x16 (pub __m512);

impl F32x16 {
    pub const WIDTH: usize = 16;

    pub fn zero() -> F32x16 {
        F32x16::splat(0.0)
    }

    pub fn splat(scalar: f32) -> F32x16 {
        unsafe { F32x16(_mm512_set1_ps(scalar)) }
    }

    pub fn splat_f32x4(v: F32x4) -> F32x16 {
        unsafe { F32x16(_mm512_broadcast_f32x4(v.0)) }
    }

    pub fn min<T: Into<Self>>(self, b: T) -> F32x16 {
        unsafe { F32x16(_mm512_min_ps(self.0, b.into().0)) }
    }

    pub fn max<T: Into<Self>>(self, b: T) -> F32x16 {
        unsafe { F32x16(_mm512_max_ps(self.0, b.into().0)) }
    }

    pub fn clamp<T: Into<Self>>(self, min: T, max: T) -> F32x16 {
        Self::max(min.into(), Self::min(self, max))
    }

    pub fn abs(self) -> F32x16 {
        unsafe { F32x16(_mm512_abs_ps(self.0)) }
    }

    pub fn cmp_lt<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OS>(self.0, b.into().0) }
    }

    pub fn cmp_eq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OS>(self.0, b.into().0) }
    }
}

impl Sqrt for F32x16 {
    type Output = F32x16;
    fn sqrt(self) -> Self::Output {
        unsafe { F32x16(_mm512_sqrt_ps(self.0)) }
    }
}

impl From<f32> for F32x16 {
    fn from(scalar: f32) -> F32x16 {
        F32x16::splat(scalar)
    }
}

impl From<std::simd::u32x16> for F32x16 {
    fn from(v: std::simd::u32x16) -> F32x16 {
        unsafe { F32x16(_mm512_cvtepu32_ps(v.into())) }
    }
}

impl From<__m512> for F32x16 {
    fn from(v: __m512) -> F32x16 {
        unsafe { F32x16(v) }
    }
}

impl Into<[f32; 16]> for F32x16 {
    fn into(self) -> [f32; 16] {
        let mut array = [0f32; 16];
        unsafe { _mm512_store_ps(array.as_mut_ptr(), self.0); }
        return array;
    }
} 

impl From<&[f32; 16]> for F32x16 {
    fn from(v: &[f32; 16]) -> F32x16 {
        unsafe { F32x16(_mm512_loadu_ps(v.as_ptr())) }
    }
}

impl Add<F32x16> for F32x16 {
    type Output = F32x16;
    fn add(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_add_ps(self.0, _rhs.0)) }
    }
}

impl Add<f32> for F32x16 {
    type Output = F32x16;
    fn add(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_add_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Sub<F32x16> for F32x16 {
    type Output = F32x16;
    fn sub(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_sub_ps(self.0, _rhs.0)) }
    }
}

impl Sub<f32> for F32x16 {
    type Output = F32x16;
    fn sub(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_sub_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Mul<F32x16> for F32x16 {
    type Output = F32x16;
    fn mul(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_mul_ps(self.0, _rhs.0)) }
    }
}

impl Mul<f32> for F32x16 {
    type Output = F32x16;
    fn mul(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_mul_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Div<F32x16> for F32x16 {
    type Output = F32x16;
    fn div(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_div_ps(self.0, _rhs.0)) }
    }
}

impl Div<f32> for F32x16 {
    type Output = F32x16;
    fn div(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_div_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

pub struct F32x4 (pub __m128);

impl From<&Bgraf32> for F32x4 {
    fn from(bgraf32: &Bgraf32) -> F32x4 {
        unsafe { F32x4(_mm_loadu_ps(bgraf32.val.as_ptr()) )}
    }
}

impl From<f32> for F32x4 {
    fn from(scalar: f32) -> F32x4 {
        unsafe { F32x4(_mm_set1_ps(scalar)) }
    }   
}

impl From<std::simd::u32x4> for F32x4 {
    fn from(v: std::simd::u32x4) -> F32x4 {
        unsafe { F32x4(_mm_cvtepi32_ps(v.into())) }
    }
}

impl From<&[f32; 4]> for F32x4 {
    fn from(v: &[f32; 4]) -> F32x4 {
        unsafe { F32x4(_mm_loadu_ps(v.as_ptr())) }
    }
}

impl Add<F32x4> for F32x4 {
    type Output = F32x4;
    fn add(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_add_ps(self.0, _rhs.0)) }
    }
}

impl Add<f32> for F32x4 {
    type Output = F32x4;
    fn add(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_add_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Sub<F32x4> for F32x4 {
    type Output = F32x4;
    fn sub(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_sub_ps(self.0, _rhs.0)) }
    }
}

impl Sub<f32> for F32x4 {
    type Output = F32x4;
    fn sub(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_sub_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Mul<F32x4> for F32x4 {
    type Output = F32x4;
    fn mul(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_mul_ps(self.0, _rhs.0)) }
    }
}

impl Mul<f32> for F32x4 {
    type Output = F32x4;
    fn mul(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_mul_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

impl Div<F32x4> for F32x4 {
    type Output = F32x4;
    fn div(self, _rhs: F32x4) -> F32x4 {
        unsafe { F32x4(_mm_div_ps(self.0, _rhs.0)) }
    }
}

impl Div<f32> for F32x4 {
    type Output = F32x4;
    fn div(self, _rhs: f32) -> F32x4 {
        unsafe { F32x4(_mm_div_ps(self.0, _mm_set1_ps(_rhs))) }
    }
}

