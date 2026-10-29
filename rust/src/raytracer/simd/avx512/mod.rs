/// My custom AVX512 intrinsics wrapper.

mod f32x4;
mod f32x16;
mod u8x16;
mod u32x16;

pub use f32x4::F32x4;
pub use f32x16::F32x16;
pub use u8x16::U8x16;
pub use u32x16::U32x16;