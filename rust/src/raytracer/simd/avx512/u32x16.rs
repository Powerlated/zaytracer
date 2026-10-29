use std::{arch::x86_64::*, ops::*};

#[derive(Clone, Copy)]
pub struct U32x16 (pub __m512i);

impl U32x16 {
    pub const WIDTH: usize = 16;

    #[inline] pub fn splat(scalar: u32) -> U32x16 {
        unsafe { U32x16(_mm512_set1_epi32(scalar as i32)) }
    }

    #[inline] pub fn cmp_eq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask::<_CMP_EQ_OQ>(self.0, b.into().0) }
    }

    #[inline] pub fn cmp_neq<T: Into<Self>>(self, b: T) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask::<_CMP_NEQ_UQ>(self.0, b.into().0) }
    }

    // #[inline] pub fn min<T: Into<Self>>(self, b: T) -> U32x16 {
    //     unsafe { U32x16(_mm512_min_ps(self.0, b.into().0)) }
    // }

    // #[inline] pub fn max<T: Into<Self>>(self, b: T) -> U32x16 {
    //     unsafe { U32x16(_mm512_max_ps(self.0, b.into().0)) }
    // }

    // #[inline] pub fn clamp<T: Into<Self>>(self, min: T, max: T) -> U32x16 {
    //     Self::max(min.into(), Self::min(self, max))
    // }
}

impl From<u32> for U32x16 {
    #[inline] fn from(scalar: u32) -> U32x16 {
        Self::splat(scalar)
    }
}

impl From<std::simd::u32x16> for U32x16 {
    #[inline] fn from(v: std::simd::u32x16) -> U32x16 {
        U32x16(v.into())
    }
}

impl From<__m512i> for U32x16 {
    #[inline] fn from(v: __m512i) -> U32x16 {
        Self(v)
    }
}

impl Into<[u32; 16]> for U32x16 {
    #[inline] fn into(self) -> [u32; 16] {
        let mut array = [0u32; 16];
        unsafe { _mm512_store_epi32(array.as_mut_ptr() as *mut i32, self.0); }
        return array;
    }
} 

impl From<&[u32; 16]> for U32x16 {
    #[inline] fn from(v: &[u32; 16]) -> U32x16 {
        unsafe { U32x16(_mm512_loadu_epi32(v.as_ptr() as *const i32)) }
    }
}

impl Add<U32x16> for U32x16 {
    type Output = U32x16;
    #[inline] fn add(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_add_epi32(self.0, _rhs.0)) }
    }
}

impl Add<u32> for U32x16 {
    type Output = U32x16;
    #[inline] fn add(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_add_epi32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

impl Sub<U32x16> for U32x16 {
    type Output = U32x16;
    #[inline] fn sub(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_sub_epi32(self.0, _rhs.0)) }
    }
}

impl Sub<u32> for U32x16 {
    type Output = U32x16;
    #[inline] fn sub(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_sub_epi32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

impl Mul<U32x16> for U32x16 {
    type Output = U32x16;
    #[inline] fn mul(self, _rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_mul_epu32(self.0, _rhs.0)) }
    }
}

impl Mul<u32> for U32x16 {
    type Output = U32x16;
    #[inline] fn mul(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_mul_epu32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}
