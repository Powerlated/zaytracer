use crate::raytracer::scalar::math::Bgra;

#[derive(Clone)]
pub struct MaterialShader {
    pub color: Bgra,
    pub emission: f32,
}

impl MaterialShader {
    pub fn new_diffuse(r: f32, g: f32, b: f32) -> MaterialShader {
        return Self { color: Bgra::from_rgb_f32(r, g, b), emission: 0.0 };
    }

    pub fn new_emissive(r: f32, g: f32, b: f32, emission: f32) -> MaterialShader {
        return Self { color: Bgra::from_rgb_f32(r, g, b), emission: emission };
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
