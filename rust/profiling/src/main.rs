#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use raytracer::raytracer::{HEIGHT, RaytracerBuf, WIDTH};
use raytracer::raytracer::scalar::RaytracerScalar;
use raytracer::raytracer::simd::RaytracerSimd;

fn main() {
    let mut buffer_boxed_slice = vec![0u8; WIDTH * 4 * HEIGHT].into_boxed_slice();
    let unboxed_buffer = &mut *buffer_boxed_slice;
    let buffer: &mut RaytracerBuf = unboxed_buffer.try_into().unwrap();

    // Set this thread to high priority
    thread_priority::set_winapi_thread_priority(
        thread_priority::thread_native_id(), 
        thread_priority::WinAPIThreadPriority::Highest
    ).unwrap();

    let raytracer = RaytracerScalar::new_with_seed(0);

    #[cfg(not(feature = "no_simd"))]
    let mut raytracer = RaytracerSimd::<14>::from_scalar(raytracer);

    // Warm up the cache
    for _ in 0..16 {
        raytracer.raytrace_step(std::hint::black_box(buffer));
    }
    unsafe { amdProfileResumeImpl(); }
    raytracer.raytrace_step(std::hint::black_box(buffer));
    unsafe { amdProfilePauseImpl(); }

    let mut cumulative_sum: u64 = 0;
    for d in buffer_boxed_slice {
        cumulative_sum += d as u64;
    }
    println!("Sum of all numbers in the ray tracer output buffer: {}", cumulative_sum);
}