use std::thread;
use std::time::Duration;

use profiling::{amdProfilePauseImpl, amdProfileResumeImpl};

use raytracer::{WINDOW_HEIGHT, WINDOW_WIDTH};
use raytracer::raytracer::Raytracer;

fn main() {
    let mut buffer_orig = vec![0u8; WINDOW_WIDTH * 4 * WINDOW_HEIGHT].into_boxed_slice();

    // Cast buffer into properly typed 2D array
    let u32_slice: &mut [u32] = bytemuck::cast_slice_mut(&mut buffer_orig);
    let flat_array_ref: &mut [u32; WINDOW_WIDTH * WINDOW_HEIGHT] = u32_slice.try_into().unwrap();
    let buffer: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT] = bytemuck::cast_mut(flat_array_ref);

    let mut raytracer = Raytracer::new();
    unsafe { amdProfileResumeImpl(); }
    raytracer.raytrace_step(std::hint::black_box(buffer));
    unsafe { amdProfilePauseImpl(); }

    let mut cumulative_sum: u64 = 0;
    for d in buffer_orig {
        cumulative_sum += d as u64;
    }
    println!("Sum of all numbers in the ray tracer output buffer: {}", cumulative_sum);
}