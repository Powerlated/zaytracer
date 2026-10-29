#![feature(portable_simd)]
#![feature(explicit_tail_calls)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(pointer_is_aligned_to)]

include!(concat!(env!("OUT_DIR"), "/generated_data.rs"));
pub mod raytracer;
