#![feature(portable_simd)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

include!(concat!(env!("OUT_DIR"), "/generated_data.rs"));

pub mod raytracer;
pub mod scene;
pub mod math;
pub mod shader;

pub use raytracer::*;
pub use scene::*;
pub use math::*;
pub use shader::*;