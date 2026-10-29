use std::{arch::x86_64::*, ops::*};

use crate::raytracer::{math::*, simd::avx512::{F32x4, U32x16}};

#[derive(Clone, Copy, Debug)]
pub struct F32x16 (pub __m512);

impl F32x16 {
    pub const WIDTH: usize = 16;

    #[inline] pub fn from_array(v: &[f32; 16]) -> Self {
        unsafe { Self(_mm512_loadu_ps(v.as_ptr())) }
    }

    #[inline] pub unsafe fn load(slice: *const f32) -> F32x16 {
        unsafe { F32x16(_mm512_loadu_ps(slice)) }
    }

    #[inline] pub unsafe fn store(slice: *mut f32, v: Self) {
        unsafe { _mm512_storeu_ps(slice, v.0) }
    }

    #[inline] pub fn gather<const SCALE: i32>(slice: *const f32, offsets: U32x16) -> F32x16 {
        unsafe { F32x16(_mm512_i32gather_ps::<SCALE>(offsets.0, slice)) }
    }

    #[inline] pub fn zero() -> F32x16 {
        F32x16::splat(0.0)
    }

    #[inline] pub fn splat(scalar: f32) -> F32x16 {
        unsafe { F32x16(_mm512_set1_ps(scalar)) }
    }

    #[inline] pub fn splat_f32x4(v: F32x4) -> F32x16 {
        unsafe { F32x16(_mm512_broadcast_f32x4(v.0)) }
    }

    #[inline] pub fn min<T: Into<Self>>(self, b: T) -> F32x16 {
        unsafe { F32x16(_mm512_min_ps(self.0, b.into().0)) }
    }

    #[inline] pub fn max<T: Into<Self>>(self, b: T) -> F32x16 {
        unsafe { F32x16(_mm512_max_ps(self.0, b.into().0)) }
    }

    #[inline] pub fn clamp<T: Into<Self>>(self, min: T, max: T) -> F32x16 {
        Self::max(min.into(), Self::min(self, max))
    }

    #[inline] pub fn abs(self) -> F32x16 {
        unsafe { F32x16(_mm512_abs_ps(self.0)) }
    }

    #[inline] pub fn cmp_lt<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_gt<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_GT_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_le<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_LE_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_ge<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_GE_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_eq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_neq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_OS>(self.0, b.into().0) }
    }

    #[inline] pub fn rcp14(self) -> Self {
        unsafe { Self(_mm512_rcp14_ps(self.0)) }
    }

    #[inline] pub unsafe fn compressstoreu(self, slice: *mut f32, mask: __mmask16) {
        // compressstoreu is slow on Zen 4, use mask and then store
        unsafe {
            let store_mask = 0xFFFF >> mask.count_zeros();
            _mm512_mask_storeu_ps(slice, store_mask, _mm512_mask_compress_ps(self.0, mask, self.0));
        }
    }

    #[inline] pub unsafe fn mask(self, mask: __mmask16) -> F32x16M {
        F32x16M(self.0, mask)
    }
}

impl MulAdd for F32x16 {
    #[inline] fn mul_add(self, b: Self, c: Self) -> Self {
        unsafe { Self(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }
}

impl MulSub for F32x16 {
    #[inline] fn mul_sub(self, b: Self, c: Self) -> Self {
        unsafe { Self(_mm512_fmsub_ps(self.0, b.0, c.0)) }
    }
}

impl Sqrt for F32x16 {
    type Output = F32x16;
    #[inline] fn sqrt(self) -> Self::Output {
        unsafe { F32x16(_mm512_sqrt_ps(self.0)) }
    }
}

impl From<f32> for F32x16 {
    #[inline] fn from(scalar: f32) -> F32x16 {
        F32x16::splat(scalar)
    }
}

impl From<std::simd::u32x16> for F32x16 {
    #[inline] fn from(v: std::simd::u32x16) -> F32x16 {
        unsafe { F32x16(_mm512_cvtepu32_ps(v.into())) }
    }
}

impl From<U32x16> for F32x16 {
    #[inline] fn from(v: U32x16) -> F32x16 {
        unsafe { F32x16(_mm512_cvtepu32_ps(v.0)) }
    }
}

impl From<__m512> for F32x16 {
    #[inline] fn from(v: __m512) -> F32x16 {
        unsafe { F32x16(v) }
    }
}

impl Into<[f32; 16]> for F32x16 {
    #[inline] fn into(self) -> [f32; 16] {
        let mut array = [0f32; 16];
        unsafe { _mm512_store_ps(array.as_mut_ptr(), self.0); }
        return array;
    }
} 

impl From<&[f32; 16]> for F32x16 {
    #[inline] fn from(v: &[f32; 16]) -> Self {
        Self::from_array(v)
    }
}

impl Add<F32x16> for F32x16 {
    type Output = F32x16;
    #[inline] fn add(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_add_ps(self.0, _rhs.0)) }
    }
}

impl Add<f32> for F32x16 {
    type Output = F32x16;
    #[inline] fn add(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_add_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Sub<F32x16> for F32x16 {
    type Output = F32x16;
    #[inline] fn sub(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_sub_ps(self.0, _rhs.0)) }
    }
}

impl Sub<f32> for F32x16 {
    type Output = F32x16;
    #[inline] fn sub(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_sub_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Mul<F32x16> for F32x16 {
    type Output = F32x16;
    #[inline] fn mul(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_mul_ps(self.0, _rhs.0)) }
    }
}

impl Mul<f32> for F32x16 {
    type Output = F32x16;
    #[inline] fn mul(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_mul_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

impl Div<F32x16> for F32x16 {
    type Output = F32x16;
    #[inline] fn div(self, _rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_div_ps(self.0, _rhs.0)) }
    }
}

impl Div<f32> for F32x16 {
    type Output = F32x16;
    #[inline] fn div(self, _rhs: f32) -> F32x16 {
        unsafe { F32x16(_mm512_div_ps(self.0, _mm512_set1_ps(_rhs))) }
    }
}

pub struct F32x16M (pub __m512, pub __mmask16);

impl F32x16M {
    #[inline] pub fn splat(self, scalar: f32) -> F32x16 {
        unsafe { F32x16(_mm512_mask_mov_ps(self.0, self.1, _mm512_set1_ps(scalar))) }
    }

    #[inline] pub fn mul(self, rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_mask_mul_ps(self.0, self.1, rhs.0, self.0)) }
    }

    #[inline] pub fn xor(self, rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_mask_xor_ps(self.0, self.1, rhs.0, self.0)) }
    }

    #[inline] pub fn mov(self, rhs: F32x16) -> F32x16 {
        unsafe { F32x16(_mm512_mask_mov_ps(self.0, self.1, rhs.0)) }
    }

    #[inline] pub fn scatter<const SCALE: i32>(self, slice: *mut f32, offsets: U32x16) {
        unsafe { _mm512_mask_i32scatter_ps::<SCALE>(slice, self.1, offsets.0, self.0) }
    }
}