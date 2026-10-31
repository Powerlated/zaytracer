use std::ops::*;

pub struct SFC32<T> {
    a: T,
    b: T,
    c: T,
    d: T,
}

impl<T> SFC32<T>
where 
    T: Add<Output = T> + Sub<Output = T>
     + Shl<u8, Output = T> + Shr<u8, Output = T>
     + BitOr<Output = T> + BitXor<Output = T>
     + Clone + Copy
     + From<u32>
{
    pub fn seed(seed: T) -> Self {
        let mut sfc32 = Self {
            a: 0.into(),
            b: seed,
            c: 0.into(),
            d: 1.into(),
        };

        for _ in 0..24 {
            sfc32.next_u32();
        }

        sfc32
    }

    pub fn next_u32(&mut self) -> T {
        let barrel_shift: u8 = 21u8;
        let rshift: u8 = 9u8;
        let lshift: u8 = 3u8;

        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;

        let result = a + b + d;

        self.a = b ^ (b >> rshift);
        self.b = c + (c << lshift);
        self.c = ((c << barrel_shift) | (c >> (32u8 - barrel_shift))) + result;
        self.d = d + 1.into();

        result
    }
}