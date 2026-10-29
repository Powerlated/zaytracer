#![feature(portable_simd)]
#![feature(explicit_tail_calls)]
#![feature(pointer_is_aligned_to)]

include!(concat!(env!("OUT_DIR"), "/generated_data.rs"));
pub mod raytracer;
