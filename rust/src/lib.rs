#![feature(portable_simd)]
#![feature(explicit_tail_calls)]

include!(concat!(env!("OUT_DIR"), "/generated_data.rs"));
pub mod raytracer;
