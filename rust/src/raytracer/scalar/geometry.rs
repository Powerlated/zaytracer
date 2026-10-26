use crate::raytracer::scalar::{MaterialShader, TriShader, math::{Mat33, Mat44, Vec3}, shader::TriPerVertexShader};

#[derive(Clone)]
pub enum Geo {
    Tri(Tri),
    // sphere: Sphere,
}

pub enum IntersectionGeo<'a> {
    Tri(TriIntersection<'a>),
    // sphere: null
}

pub struct Intersection<'a> {
    pub p: Vec3, // Point of intersection
    pub n: Vec3, // Normalized normal vector of intersection surface
    pub t: f32, // t value of intersection
    pub specific: IntersectionGeo<'a>, 
}

#[derive(Clone)]
pub struct Tri {
    // Vertices. typical notation in math is (A, B, C)
    pub v: [Vec3; 3],
    pub shader: TriShader,
    // TODO: derive normal direction from winding order?
    // n: Vec3, // n^\hat is normalized normal vector
}

#[derive(Clone)]
pub struct TriIntersection<'a> {
    pub w: f32,
    pub u: f32,
    pub v: f32,
    pub tri: &'a Tri,
}

impl Tri {
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    // `O`: Origin point
    // `D`: Direction vector
    #[allow(non_snake_case)]
    pub fn intersect(self: &Tri, O: Vec3, D: Vec3) -> Option<(Intersection<'_>, TriIntersection<'_>)> {
        let v1 = self.v[0];
        let v2 = self.v[1];
        let v3 = self.v[2];

        // **Obtain normal vector of triangle**
        // TODO: use stored n instead of calculating on the fly
        let n = (v2.sub(v1)).cross(v3.sub(v1));

        // **Check if the ray is parallel to the triangle**
        // when ray direction vector (D) is parallel to plane (i.e. D.dot(n) == 0)
        // ray does not intersect with plane at all
        const EPSILON: f32 = 1e-6;
        if f32::abs(D.dot(n)) < EPSILON {
            return None;
        }

        // **Check if the ray-plane intersection lies outside of the triangle**
        // Find intersection point on triangle in barycentric coordinates
        //      P = w*v1 + u*v2 + v*v3;
        // P is a convex combination of the triangle's vertices, meaning that
        // the coefficients w, u, v, must sum to 1. Because of this, we can
        // replace w with 1 - u - v.
        //      P = (1 - u - v)v1 + u*v2 + v*v3
        //
        // This can be massaged into a system of linear equations:
        //      O - v1 = -t*D + u*(v2 - v1) + v*(v3 - v1);
        // Keep in mind -D, v1, v2, v3, are 3-entry vectors, so we can write
        // this system in matrix-vector multiplication form.
        //
        // [ -D_x  (v2-v1)_x  (v3-v1)_x ][ t ]   [ O_x - v1_x ]
        // [ -D_y  (v2-v1)_y  (v3-v1)_y ][ u ] = [ O_y - v1_y ]
        // [ -D_z  (v2-v1)_z  (v3-v1)_z ][ v ]   [ O_z - v1_z ]
        //
        // https://en.wikipedia.org/wiki/Cramer%27s_rule
        // Cramer's rule is an explicit formula for the solution of a system of
        // linear equations with as many equations as unknown, valid whenever
        // the system has a unique solution.
        //
        // Cramer's rule:
        // Consider a typical system of n linear equations for n unknowns,
        // represented in typical matrix multiplication form:
        //      Ax = b
        // x_i is the column vector of the solutions
        //      x_i = det(A_i) / det(A)
        // where A_i is the matrix formed by replacing the i-th column of A by
        // the column vector b

        let A = Mat33::from_col_vecs(D.mul_scalar(-1.0), v2.sub(v1), v3.sub(v1));
        let b = O.sub(v1);

        // t=x_1, u=x_2, v=x_3
        let t = A.det_with_col_replaced::<0>(b) / A.det();
        let u = A.det_with_col_replaced::<1>(b) / A.det();
        let v = A.det_with_col_replaced::<2>(b) / A.det();

        // Return None if intersection of line with triangle is *behind* the direction vector
        if t < 0. {
            return None;
        }

        // Recover w
        let w = 1. - u - v;

        // Return None if intersection of line with triangle is not inside triangle
        if w < 0. || u < 0. || v < 0. {
            return None;
        }

        // Obtain intersection point, from the definition of barycentric coordinates
        let P = v1.mul_scalar(w).add(v2.mul_scalar(u)).add(v3.mul_scalar(v));

        let tri_inter = TriIntersection {w, u, v, tri: &self};
        return Some((Intersection { 
            p: P, n: n.normalize(), t: t,
            specific: IntersectionGeo::Tri(tri_inter.clone()),
        }, tri_inter)); 
    }
}

#[cfg(test)]
mod test {
    use crate::raytracer::scalar::{geometry::Tri, math::Vec3, shader::{MaterialShader, TriShader}};

    const EPSILON: f32 = 1e-5;
    fn expect_intersect(tri: Tri, o: Vec3, d: Vec3, intersect_pos: [f32; 3], t: f32) {
        let maybe_ti = tri.intersect(o, d);
        assert!(maybe_ti.is_some());
        if let Some(ti) = maybe_ti {
            assert!((t-ti.0.t).abs() <= EPSILON);
            for i in 0..3 {
                assert!((ti.0.p.val[i]-intersect_pos[i]).abs() <= EPSILON);
            }
        }
    }
    
    fn expect_no_intersect(tri: Tri, o: Vec3, d: Vec3)  {
        let maybe_ti = tri.intersect(o, d);
        // if (maybe_ti) |ti| {
            // std.debug.print("Intersection point at [{} {} {}]", .{ ti.p.val[0], ti.p.val[1], ti.p.val[2] });
        // }
        assert!(maybe_ti.is_none());
    }

    fn basic_shader() -> TriShader {
        TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))
    }

    #[test]
    fn intersection_1() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-1., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: basic_shader() };

        expect_intersect(tri, ray_pos, ray_dir, [0., 0., 1.], 1.);
    }

    #[test]
    fn intersection_2() {
        let ray_pos = Vec3::from_xyz(5., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-10., -1., 1.), Vec3::from_xyz(10., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: basic_shader() };

        expect_intersect(tri, ray_pos, ray_dir, [5., 0., 1.], 1.);
    }

    #[test]
    fn intersection_3() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(1., 0., 1.).normalize();

        let tri = Tri { v: [Vec3::from_xyz(-10., -100., 1.), Vec3::from_xyz(100., -100., 1.), Vec3::from_xyz(0., 100., 1.)], shader: basic_shader() };

        let tri2 = Tri { v: [Vec3::from_xyz(-10., -100., 2.), Vec3::from_xyz(100., -100., 2.), Vec3::from_xyz(0., 100., 2.)], shader: basic_shader() };

        expect_intersect(tri, ray_pos, ray_dir, [1., 0., 1.], f32::sqrt(2.));
        expect_intersect(tri2, ray_pos, ray_dir, [2., 0., 2.], f32::sqrt(2.));
    }

    #[test]
    fn non_intersection_1() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 1., -1.);

        let tri = Tri { v: [Vec3::from_xyz(-1., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: basic_shader() };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }

    #[test]
    fn non_intersection_2() {
        let ray_pos = Vec3::from_xyz(5., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-10., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: basic_shader() };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }

    #[test]
    fn non_intersection_3() {
        let ray_pos = Vec3::from_xyz(1., -1., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(0.5, 0.5, 1.), Vec3::from_xyz(-0.5, 0.5, 1.), Vec3::from_xyz(0., -0.5, 1.)], shader: basic_shader() };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }
}

struct Sphere {
    center: Vec3, // typical notation in math is (h, k, j)
    radius: f32,
}

fn trongle() -> Geo {
    Geo::Tri(Tri {
            v: [
                Vec3::from_xyz(0.5, -0.5, 1.),
                Vec3::from_xyz(-0.5, -0.5, 1.),
                Vec3::from_xyz(0., 0.5, 1.)
            ],
            shader: TriShader::PerVertex(
                TriPerVertexShader { colors: [
                    [ 1., 0., 0., 0.5 ],
                    [ 0., 1., 0., 0.5 ],
                    [ 0., 0., 1., 0.5 ]
                ]}
            )
        },
    )
}

/// Square with side length `len`, centered on (0, 0, 0), normal facing (0, 0, 1).
pub fn square(len: f32, shader: TriShader) -> [Geo; 2] {
    let half_len = len;

    let t1 = Tri {
        v: [
            Vec3::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz(-half_len / 2.,  half_len / 2., 0.),
            Vec3::from_xyz( half_len / 2.,  half_len / 2., 0.)
        ],
        shader: shader.clone(),
    };

    let t2 = Tri {
        v: [
            Vec3::from_xyz( half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz( half_len / 2.,  half_len / 2., 0.),
        ],
        shader: shader.clone(),
    };

    let g1 = Geo::Tri(t1);
    let g2 = Geo::Tri(t2);

    return [g1, g2];
}
    
/// Cube with side length `len`, centered on (0, 0, 0).
pub fn cube(len: f32) -> [Geo; 12] {
    let f0 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  x
    let f1 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  y
    let f2 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  z
    let f3 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -x
    let f4 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -y
    let f5 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -z   

    let mut faces = [
        f0[0].clone(), f0[1].clone(),
        f1[0].clone(), f1[1].clone(),
        f2[0].clone(), f2[1].clone(),
        f3[0].clone(), f3[1].clone(),
        f4[0].clone(), f4[1].clone(),
        f5[0].clone(), f5[1].clone()
    ];

    transform_in_place(&mut faces[0..2  ], Mat33::rotation(  0., -90.,  0.).translate( len/2.,      0.,      0.)); //  x
    transform_in_place(&mut faces[2..4  ], Mat33::rotation(-90.,   0.,  0.).translate(     0.,  len/2.,      0.)); //  y
    transform_in_place(&mut faces[4..6  ], Mat33::rotation(  0.,   0.,  0.).translate(     0.,      0.,  len/2.)); //  z
    transform_in_place(&mut faces[6..8  ], Mat33::rotation(  0.,  90.,  0.).translate(-len/2.,      0.,      0.)); // -x
    transform_in_place(&mut faces[8..10 ], Mat33::rotation( 90.,   0.,  0.).translate(     0., -len/2.,      0.)); // -y
    transform_in_place(&mut faces[10..12], Mat33::rotation(180.,   0.,  0.).translate(     0.,      0., -len/2.)); // -z

    return faces;
}

pub fn transform_in_place(geos: &mut [Geo], transformation: Mat44) {
    for geo in geos {
        match geo {
            Geo::Tri(tri) => {
                // apply homogenous transformation
                let a = transformation.val;
                for v in tri.v.iter_mut() {
                    let b = [v.val[0], v.val[1], v.val[2], 1.];

                    let mut vec_val = [0.0; 4];
                    for j in 0..4 {
                        vec_val[j] =
                            a[j * 4 + 0] * b[0] +
                            a[j * 4 + 1] * b[1] +
                            a[j * 4 + 2] * b[2] +
                            a[j * 4 + 3] * b[3];
                    }

                    *v = Vec3::from_xyz(vec_val[0], vec_val[1], vec_val[2]);
                }
            },
            _ => unimplemented!()
        }
    }
}