use std::time::Duration;

use raytracer::raytracer::scalar::RaytracerScalar;
use raytracer::raytracer::simd::RaytracerSimd;
use raytracer::raytracer::RaytracerBuf;
use raytracer::raytracer::WIDTH;
use raytracer::raytracer::HEIGHT;

fn main() {
    // Move this thread onto the highest-numbered logical core
    let core_ids = core_affinity::get_core_ids().unwrap();
    core_affinity::set_for_current(*core_ids.last().unwrap());
    
    // Set this thread to high priority
    thread_priority::set_winapi_thread_priority(
        thread_priority::thread_native_id(), 
        thread_priority::WinAPIThreadPriority::Highest
    ).unwrap();

    let mut buffer_boxed_slice = vec![0u8; WIDTH * 4 * HEIGHT].into_boxed_slice();
    let unboxed_buffer = &mut *buffer_boxed_slice;
    let buf: &mut RaytracerBuf = unboxed_buffer.try_into().unwrap();

    println!("Starting benchmark...");

    let mut times = [Duration::ZERO; 16];

    const ITERS: usize = 5;
    for _ in 0..ITERS {
        let mut tracer0 = RaytracerSimd::<0>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer1 = RaytracerSimd::<1>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer2 = RaytracerSimd::<2>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer3 = RaytracerSimd::<3>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer4 = RaytracerSimd::<4>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer5 = RaytracerSimd::<5>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer6 = RaytracerSimd::<6>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer7 = RaytracerSimd::<7>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer8 = RaytracerSimd::<8>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer9 = RaytracerSimd::<9>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer10 = RaytracerSimd::<10>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer11 = RaytracerSimd::<11>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer12 = RaytracerSimd::<12>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer13 = RaytracerSimd::<13>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer14 = RaytracerSimd::<14>::from_scalar(RaytracerScalar::new_with_seed(0));
        let mut tracer15 = RaytracerSimd::<15>::from_scalar(RaytracerScalar::new_with_seed(0));

        // Run in forward through the tracers
        for i in 0..16usize {
            let start_time = std::time::Instant::now();
            match i {
                0 => { tracer0.raytrace_to_completion(buf); }
                1 => { tracer1.raytrace_to_completion(buf); }
                2 => { tracer2.raytrace_to_completion(buf); }
                3 => { tracer3.raytrace_to_completion(buf); }
                4 => { tracer4.raytrace_to_completion(buf); }
                5 => { tracer5.raytrace_to_completion(buf); }
                6 => { tracer6.raytrace_to_completion(buf); }
                7 => { tracer7.raytrace_to_completion(buf); }
                8 => { tracer8.raytrace_to_completion(buf); }
                9 => { tracer9.raytrace_to_completion(buf); }
                10 => { tracer10.raytrace_to_completion(buf); }
                11 => { tracer11.raytrace_to_completion(buf); }
                12 => { tracer12.raytrace_to_completion(buf); }
                13 => { tracer13.raytrace_to_completion(buf); }
                14 => { tracer14.raytrace_to_completion(buf); }
                15 => { tracer15.raytrace_to_completion(buf); }
                16_usize.. => todo!()
            }
            let elapsed = start_time.elapsed();
            times[i] += elapsed;
        } 

        // Run in reverse through the tracers
        for i in (0..16usize).rev() {
            let start_time = std::time::Instant::now();
            match i {
                0 => { tracer0.raytrace_to_completion(buf); }
                1 => { tracer1.raytrace_to_completion(buf); }
                2 => { tracer2.raytrace_to_completion(buf); }
                3 => { tracer3.raytrace_to_completion(buf); }
                4 => { tracer4.raytrace_to_completion(buf); }
                5 => { tracer5.raytrace_to_completion(buf); }
                6 => { tracer6.raytrace_to_completion(buf); }
                7 => { tracer7.raytrace_to_completion(buf); }
                8 => { tracer8.raytrace_to_completion(buf); }
                9 => { tracer9.raytrace_to_completion(buf); }
                10 => { tracer10.raytrace_to_completion(buf); }
                11 => { tracer11.raytrace_to_completion(buf); }
                12 => { tracer12.raytrace_to_completion(buf); }
                13 => { tracer13.raytrace_to_completion(buf); }
                14 => { tracer14.raytrace_to_completion(buf); }
                15 => { tracer15.raytrace_to_completion(buf); }
                16_usize.. => todo!()
            }
            let elapsed = start_time.elapsed();
            times[i] += elapsed;
        } 
    }
    
    let mut times_sorted = times.iter().enumerate().collect::<Vec<_>>();;
    times_sorted.sort_by_key(|&(_, v)| v);

    println!("Final cumulative times, in order of best to worst:");
    for (i, times) in times_sorted {
        println!("{:>2}: {} ms", i, times.as_millis());
    }

}