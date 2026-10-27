use bytemuck::{Pod, Zeroable};

use crate::raytracer::math::Bgraf32;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialShader {
    pub color: Bgraf32,
    pub emission: f32,
}

impl MaterialShader {
    pub fn new_diffuse(r: f32, g: f32, b: f32) -> MaterialShader {
        return Self { color: Bgraf32::from_rgb_f32(r, g, b), emission: 0.0 };
    }

    pub fn new_emissive(r: f32, g: f32, b: f32, emission: f32) -> MaterialShader {
        return Self { color: Bgraf32::from_rgb_f32(r, g, b), emission: emission };
    }
}

impl Default for MaterialShader {
    fn default() -> Self {
        Self::new_diffuse(1., 1., 1.)    
    }
}

#[derive(Clone)]
pub enum TriShader {
    PerVertex(TriPerVertexShader),
    Material(MaterialShader),
}

#[derive(Clone)]
pub struct TriPerVertexShader {
    // Per-vertex color
    pub colors: [[f32; 4]; 3],
}
