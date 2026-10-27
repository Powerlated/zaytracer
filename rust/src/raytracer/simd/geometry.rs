use crate::raytracer::{math::{Bgra, Mat33, Mat33f32, Vec3, Vec3f32}, scalar::shader::MaterialShader, simd::avx512::F32x16};

pub struct TriIntersectionSimd {
    pub p: Vec3::<F32x16>, // Point of intersection
    pub n: Vec3f32, // Normalized normal vector of intersection surface
    pub t: F32x16, // t value of intersection

    pub w: F32x16,
    pub u: F32x16,
    pub v: F32x16,

    pub bitfield_hit: u16,
}


#[derive(Clone)]
pub struct TriSimd {
    // Vertices. typical notation in math is (A, B, C)
    pub v: [Vec3f32; 3],
    pub shader: MaterialShader,
}

impl TriSimd {
    /// Test 16 rays against this triangle
    #[allow(non_snake_case)]
    pub fn intersect(&self, O: Vec3f32, D: Vec3::<F32x16>) -> TriIntersectionSimd {
        let v1 = self.v[0];
        let v2 = self.v[1];
        let v3 = self.v[2];

        let n = (v2.sub(v1)).cross(v3.sub(v1));

        const EPSILON: f32 = 1e-6;
        let mut bitfield_no_hit = D.dot(n).abs().cmp_lt(EPSILON);

        let A = Mat33::<F32x16>::from_col_vecs(D.mul_scalar(-1.0), v2.sub(v1).splat(), v3.sub(v1).splat());
        let b = O.sub(v1.into());

        let t = A.det_with_col_replaced::<0>(b.splat()) / A.det();
        let u = A.det_with_col_replaced::<1>(b.splat()) / A.det();
        let v = A.det_with_col_replaced::<2>(b.splat()) / A.det();

        bitfield_no_hit |= t.cmp_lt(0.);

        let w = F32x16::from(1.0) - u - v;
        bitfield_no_hit |= w.cmp_lt(0.);
        bitfield_no_hit |= u.cmp_lt(0.);
        bitfield_no_hit |= v.cmp_lt(0.);

        let p = v1.splat().mul_scalar(w).add(v2.splat().mul_scalar(u)).add(v3.splat().mul_scalar(v));

        TriIntersectionSimd { p, n, t, w, u, v, bitfield_hit: !bitfield_no_hit }
    }
}
