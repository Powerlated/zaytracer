pub mod scalar;
pub mod simd;

pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 480;
pub const STRIDE: usize = WIDTH * BYTES_PER_PIXEL;
pub const BYTES_PER_PIXEL: usize = 4;
pub const BUF_SIZE: usize = WIDTH * HEIGHT * BYTES_PER_PIXEL;

pub const MAX_ACCUMULATE_STEPS: u32 = 128;
pub const DITHER_DIVISOR: usize = 256;

pub type RaytracerBuf = [u8; BUF_SIZE];
