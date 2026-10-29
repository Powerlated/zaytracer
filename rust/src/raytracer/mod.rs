pub mod scalar;
pub mod simd;
pub mod math;

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

pub type RaytracerBuf = [u8; BUF_SIZE];
