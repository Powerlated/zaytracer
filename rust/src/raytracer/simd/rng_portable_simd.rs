// Copyright 2018 Developers of the Rand project.
// Copyright 2026 Brian Jia, modified for SIMD
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::convert::Infallible;
use std::{ops::{Add, Shl}, simd::{num::SimdUint, *}};

/// A xoshiro128++ random number generator.
///
/// The xoshiro128++ algorithm is not suitable for cryptographic purposes, but
/// is very fast and has excellent statistical properties.
///
/// The algorithm used here is translated from [the `xoshiro128plusplus.c`
/// reference source code](http://xoshiro.di.unimi.it/xoshiro128plusplus.c) by
/// David Blackman and Sebastiano Vigna.
/// 
pub struct Xoshiro128PlusPlusSimd {
    s: [u32x4; 4],
}

impl Xoshiro128PlusPlusSimd {
    /// Create a new `Xoshiro128PlusPlus` from a `u64` seed.
    ///
    /// This uses the SplitMix64 generator internally.
    #[inline]
    pub fn seed_from_u64(mut state: u64x4) -> Self {
        const PHI: u64x4 = u64x4::splat(0x9e3779b97f4a7c15);
        let mut s = [u32x4::default(); 4];
        for i in s.chunks_exact_mut(2) {
            state += PHI;
            let mut z = state;
            z = (z ^ (z >> 30)) * u64x4::splat(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)) * u64x4::splat(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            i[0] = z.cast::<u32>();
            i[1] = (z >> 32).cast::<u32>();
        }

        Xoshiro128PlusPlusSimd { s }
    }

    pub fn next_u32(&mut self) -> u32x4 {
        let intermed = self.s[0] + self.s[3];
        let res = ((intermed << 7) | (intermed >> (32 - 7))) + self.s[0];

        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = (self.s[3] << 11) | (self.s[3] >> (32 - 11));

        res
    }
}