use profiling::{amdProfilePauseImpl, amdProfileResumeImpl};

use raytracer::{WINDOW_HEIGHT, WINDOW_WIDTH};
use raytracer::raytracer::Raytracer;

fn main() {
    let mut buffer = vec![0u8; WINDOW_WIDTH * 4 * WINDOW_HEIGHT].into_boxed_slice();

    // Cast buffer into properly typed 2D array
    let u32_slice: &mut [u32] = bytemuck::cast_slice_mut(&mut buffer);
    let flat_array_ref: &mut [u32; WINDOW_WIDTH * WINDOW_HEIGHT] = u32_slice.try_into().unwrap();
    let buffer: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT] = bytemuck::cast_mut(flat_array_ref);

    let mut raytracer = Raytracer::new();
    unsafe { amdProfileResumeImpl(); }
    raytracer.raytrace_step(buffer);
    unsafe { amdProfilePauseImpl(); }
}