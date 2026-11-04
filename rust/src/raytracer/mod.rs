use crate::math::Bgraf32;

pub mod scalar;
pub mod simd;
pub mod rng;

pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 480;
pub const STRIDE: usize = WIDTH * BYTES_PER_PIXEL;
pub const BYTES_PER_PIXEL: usize = 4;
pub const BUF_SIZE: usize = WIDTH * HEIGHT * BYTES_PER_PIXEL;

pub const MAX_ACCUMULATE_STEPS: u32 = 128; // default
// pub const MAX_ACCUMULATE_STEPS: u32 = 1;
pub const DITHER_DIVISOR: usize = 256;
pub const MAX_BOUNCES: u8 = 16; // default
// pub const MAX_BOUNCES: u8 = 1;
pub const ENABLE_JITTER: bool = true;

pub const FORCE_HIDPI: bool = true;

pub const NUM_WORKERS_SCALAR: usize = 16;

pub type RaytracerTexBuf = [u8; WIDTH*HEIGHT*4];
pub type RaytracerBuf = [Bgraf32; WIDTH*HEIGHT];
pub fn new_raytracer_buf() -> Box<RaytracerBuf> {
    return vec![Bgraf32::zero(); WIDTH*HEIGHT].into_boxed_slice().try_into().unwrap();
}
