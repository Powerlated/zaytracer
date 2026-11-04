use std::time::Duration;

use raytracer::math::Bgraf32;
use raytracer::raytracer::MAX_ACCUMULATE_STEPS;
use raytracer::raytracer::scalar::RaytracerScalar;
use raytracer::raytracer::simd::RaytracerSimd;
use raytracer::raytracer::RaytracerBuf;
use raytracer::raytracer::WIDTH;
use raytracer::raytracer::HEIGHT;
use raytracer::scene::Scene;

fn raytrace_to_completion<const N: u32>(tracer: &mut RaytracerSimd<N>, buf: &mut RaytracerBuf) {
    for _ in 0..MAX_ACCUMULATE_STEPS {
        tracer.raytrace_step(buf);
    }
}

fn main() {
    // Move this thread onto the highest-numbered logical core
    let core_ids = core_affinity::get_core_ids().unwrap();
    core_affinity::set_for_current(*core_ids.last().unwrap());
    
    // Set this thread to high priority
    #[cfg(windows)]
    thread_priority::set_winapi_thread_priority(
        thread_priority::thread_native_id(), 
        thread_priority::WinAPIThreadPriority::Highest
    ).unwrap();

    let mut buffer_boxed_slice = vec![Bgraf32::zero(); WIDTH*HEIGHT].into_boxed_slice();
    let unboxed_buffer = &mut *buffer_boxed_slice;
    let buf: &mut RaytracerBuf = unboxed_buffer.try_into().unwrap();

    println!("Starting benchmark...");

    let mut times = [Duration::ZERO; 16];

    const ITERS: usize = 5;
    for _ in 0..ITERS {
        let mut tracer0 = RaytracerSimd::<0>::from_scene(Scene::cornell_box());
        let mut tracer1 = RaytracerSimd::<1>::from_scene(Scene::cornell_box());
        let mut tracer2 = RaytracerSimd::<2>::from_scene(Scene::cornell_box());
        let mut tracer3 = RaytracerSimd::<3>::from_scene(Scene::cornell_box());
        let mut tracer4 = RaytracerSimd::<4>::from_scene(Scene::cornell_box());
        let mut tracer5 = RaytracerSimd::<5>::from_scene(Scene::cornell_box());
        let mut tracer6 = RaytracerSimd::<6>::from_scene(Scene::cornell_box());
        let mut tracer7 = RaytracerSimd::<7>::from_scene(Scene::cornell_box());
        let mut tracer8 = RaytracerSimd::<8>::from_scene(Scene::cornell_box());
        let mut tracer9 = RaytracerSimd::<9>::from_scene(Scene::cornell_box());
        let mut tracer10 = RaytracerSimd::<10>::from_scene(Scene::cornell_box());
        let mut tracer11 = RaytracerSimd::<11>::from_scene(Scene::cornell_box());
        let mut tracer12 = RaytracerSimd::<12>::from_scene(Scene::cornell_box());
        let mut tracer13 = RaytracerSimd::<13>::from_scene(Scene::cornell_box());
        let mut tracer14 = RaytracerSimd::<14>::from_scene(Scene::cornell_box());
        let mut tracer15 = RaytracerSimd::<15>::from_scene(Scene::cornell_box());

        // Run in forward and reverse through the tracers
        let iter = (0..16usize).chain((0..16usize).rev());
        for i in iter {
            let start_time = std::time::Instant::now();
            match i {
                0 => { raytrace_to_completion(&mut tracer0, buf); }
                1 => { raytrace_to_completion(&mut tracer1, buf); }
                2 => { raytrace_to_completion(&mut tracer2, buf); }
                3 => { raytrace_to_completion(&mut tracer3, buf); }
                4 => { raytrace_to_completion(&mut tracer4, buf); }
                5 => { raytrace_to_completion(&mut tracer5, buf); }
                6 => { raytrace_to_completion(&mut tracer6, buf); }
                7 => { raytrace_to_completion(&mut tracer7, buf); }
                8 => { raytrace_to_completion(&mut tracer8, buf); }
                9 => { raytrace_to_completion(&mut tracer9, buf); }
                10 => { raytrace_to_completion(&mut tracer10, buf); }
                11 => { raytrace_to_completion(&mut tracer11, buf); }
                12 => { raytrace_to_completion(&mut tracer12, buf); }
                13 => { raytrace_to_completion(&mut tracer13, buf); }
                14 => { raytrace_to_completion(&mut tracer14, buf); }
                15 => { raytrace_to_completion(&mut tracer15, buf); }
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