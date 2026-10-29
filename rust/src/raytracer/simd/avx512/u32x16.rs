use std::{arch::x86_64::*, ops::*};

#[derive(Clone, Copy)]
pub struct U32x16 (pub __m512i);

impl U32x16 {
    pub const WIDTH: usize = 16;
    
    #[inline] pub unsafe fn load(slice: *const u32) -> U32x16 {
        unsafe { Self(_mm512_loadu_epi32(slice as *const i32)) }
    }

    #[inline] pub unsafe fn store(slice: *mut u32, v: Self) {
        unsafe { _mm512_storeu_epi32(slice as *mut i32, v.0) }
    }

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

    #[inline] pub fn compressstoreu(self, slice: *mut u32, mask: __mmask16) {
        // compressstoreu is slow on Zen 4, use mask and then store
        unsafe {
            let store_mask = 0xFFFF >> mask.count_zeros();
            _mm512_mask_storeu_epi32(slice as *mut i32, store_mask, _mm512_mask_compress_epi32(self.0, mask, self.0));
        }
    }

    #[inline] pub unsafe fn mask(self, mask: __mmask16) -> U32x16M {
        U32x16M(self.0, mask)
    }
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
        unsafe { U32x16(_mm512_mullo_epi32(self.0, _rhs.0)) }
    }
}

impl Mul<u32> for U32x16 {
    type Output = U32x16;
    #[inline] fn mul(self, _rhs: u32) -> U32x16 {
        unsafe { U32x16(_mm512_mullo_epi32(self.0, _mm512_set1_epi32(_rhs as i32))) }
    }
}

pub struct U32x16M (pub __m512i, pub __mmask16);

impl U32x16M {
    #[inline] pub fn splat(self, scalar: u32) -> U32x16 {
        unsafe { U32x16(_mm512_mask_mov_epi32(self.0, self.1, _mm512_set1_epi32(scalar as i32))) }
    }

    #[inline] pub fn mul(self, rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_mask_mullo_epi32(self.0, self.1, rhs.0, self.0)) }
    }

    #[inline] pub fn xor(self, rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_mask_xor_epi32(self.0, self.1, rhs.0, self.0)) }
    }

    #[inline] pub fn mov(self, rhs: U32x16) -> U32x16 {
        unsafe { U32x16(_mm512_mask_mov_epi32(self.0, self.1, rhs.0)) }
    }
}