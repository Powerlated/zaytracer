use std::{arch::x86_64::*, ops::*};

use crate::raytracer::simd::avx512::F32x16;

pub struct U8x16 (pub __m128i);

impl From<F32x16> for U8x16 {
    #[inline] fn from(v: F32x16) -> U8x16 {
        unsafe { U8x16(_mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(v.0))) }
    }
}

impl Into<[u8; 16]> for U8x16 {
    #[inline] fn into(self) -> [u8; 16] {
        let mut array = [0u8; 16];
        unsafe { _mm_storeu_epi8(array.as_mut_ptr() as *mut i8, self.0); }
        return array;
    }
}