use crate::raytracer::{math::{Bgra, Mat33, Mat33f32, Vec3, Vec3f32}, scalar::{geometry::Tri, shader::MaterialShader}, simd::avx512::F32x16};

pub struct TriIntersectionSimd {
    pub p: Vec3::<F32x16>, // Point of intersection
    pub t: F32x16, // t value of intersection
    
    pub w: F32x16,
    pub u: F32x16,
    pub v: F32x16,
    
    pub n: Vec3f32, // Normalized normal vector of intersection surface
    pub hitmask: u16,
}


#[derive(Clone)]
pub struct TriSimd {
    // Vertices. typical notation in math is (A, B, C)
    pub v: [Vec3f32; 3],
    pub v1_sub_v0: Vec3f32,
    pub v2_sub_v0: Vec3f32,
    pub n: Vec3f32,
}

impl TriSimd {
    /// Test 16 rays against this triangle
    #[allow(non_snake_case)]
    #[inline]
    pub fn intersect(&self, O: Vec3::<F32x16>, D: Vec3::<F32x16>) -> TriIntersectionSimd {
        let v0 = self.v[0];
        let v1 = self.v[1];
        let v2 = self.v[2];

        const EPSILON: f32 = 1e-6;
        let mut bitfield_no_hit = D.dot(self.n).abs().cmp_lt(EPSILON);

        let A = Mat33::<F32x16>::from_col_vecs(D.mul_scalar(-1.0), self.v1_sub_v0.splat(), self.v2_sub_v0.splat());
        let b = O.sub(v0.into());

        let t = A.det_with_col_replaced::<0>(b) / A.det();
        let u = A.det_with_col_replaced::<1>(b) / A.det();
        let v = A.det_with_col_replaced::<2>(b) / A.det();

        bitfield_no_hit |= t.cmp_lt(0.);

        let w = F32x16::from(1.0) - u - v;
        bitfield_no_hit |= w.cmp_lt(0.);
        bitfield_no_hit |= u.cmp_lt(0.);
        bitfield_no_hit |= v.cmp_lt(0.);

        let p = v0.splat().mul_scalar(w).add(v1.splat().mul_scalar(u)).add(v2.splat().mul_scalar(v));

        TriIntersectionSimd { p, n: self.n, t, w, u, v, hitmask: !bitfield_no_hit }
    }

    pub fn from_scalar_tri(tri: Tri) -> Self {
        let v = tri.v;
        TriSimd { 
            v,
            v1_sub_v0: v[1].sub(v[0]),
            v2_sub_v0: v[2].sub(v[0]),
            n: v[1].sub(v[0]).cross(v[2].sub(v[0])).normalize()
        }
    }
}
