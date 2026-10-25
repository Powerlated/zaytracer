use std::f32::consts::PI;
use std::ops::Div;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::raytracer::TriShader::{Material, PerVertex};
use crate::raytracer::math::{Mat33, Mat44, Rgbf32, Vec3};
mod math;

#[derive(Clone)]
struct MaterialShader {
    color: Rgbf32,
    emission: f32,
}

impl MaterialShader {
    fn new_diffuse(r: f32, g: f32, b: f32) -> MaterialShader {
        return Self { color: Rgbf32 { val: [r, g, b] }, emission: 0.0 };
    }

    fn new_emissive(r: f32, g: f32, b: f32, emission: f32) -> MaterialShader {
        return Self { color: Rgbf32 { val: [r, g, b] }, emission: emission };
    }
}

#[derive(Clone)]
enum Geo {
    Tri(Tri),
    // sphere: Sphere,
}

fn transform_in_place(geos: &mut [Geo], transformation: Mat44) {
    for geo in geos {
        match geo {
            Geo::Tri(tri) => {
                // apply homogenous transformation
                let a = transformation.val;
                for v in tri.v.iter_mut() {
                    let b = [v.val[0], v.val[1], v.val[2], 0.];

                    let mut vec_val = [0.0; 4];
                    for j in 0..4 {
                        vec_val[j] =
                            a[j * 4 + 0] * b[0] +
                            a[j * 4 + 1] * b[1] +
                            a[j * 4 + 2] * b[2];
                            a[j * 4 + 3] * b[3];
                    }

                    *v = Vec3::from_xyz(vec_val[0], vec_val[1], vec_val[2]);
                }
            },
            _ => unimplemented!()
        }
    }
}

#[derive(Clone)]
enum TriShader {
    PerVertex(TriPerVertexShader),
    Material(MaterialShader),
}

#[derive(Clone)]
struct TriPerVertexShader {
    // Per-vertex color
    colors: [[f32; 4]; 3],
}

#[derive(Clone)]
struct TriIntersection<'a> {
    w: f32,
    u: f32,
    v: f32,
    tri: &'a Tri,
}

enum IntersectionGeo<'a> {
    Tri(TriIntersection<'a>),
    // sphere: null
}

struct Intersection<'a> {
    p: Vec3, // Point of intersection
    n: Vec3, // Normalized normal vector of intersection surface
    t: f32, // t value of intersection
    specific: IntersectionGeo<'a>, 
}

#[derive(Clone)]
struct Tri {
    // Vertices. typical notation in math is (A, B, C)
    v: [Vec3; 3],
    shader: Option<TriShader>,
    // TODO: derive normal direction from winding order?
    // n: Vec3, // n^\hat is normalized normal vector
}

impl Tri {
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    // `O`: Origin point
    // `D`: Direction vector
    #[allow(non_snake_case)]
    fn intersect(self: &Tri, O: Vec3, D: Vec3) -> Option<(Intersection, TriIntersection)> {
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
    use crate::raytracer::{Tri, math::Vec3};

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

    #[test]
    fn intersection_1() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-1., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: None };

        expect_intersect(tri, ray_pos, ray_dir, [0., 0., 1.], 1.);
    }

    #[test]
    fn intersection_2() {
        let ray_pos = Vec3::from_xyz(5., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-10., -1., 1.), Vec3::from_xyz(10., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: None };

        expect_intersect(tri, ray_pos, ray_dir, [5., 0., 1.], 1.);
    }

    #[test]
    fn intersection_3() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(1., 0., 1.).normalize();

        let tri = Tri { v: [Vec3::from_xyz(-10., -100., 1.), Vec3::from_xyz(100., -100., 1.), Vec3::from_xyz(0., 100., 1.)], shader: None };

        let tri2 = Tri { v: [Vec3::from_xyz(-10., -100., 2.), Vec3::from_xyz(100., -100., 2.), Vec3::from_xyz(0., 100., 2.)], shader: None };

        expect_intersect(tri, ray_pos, ray_dir, [1., 0., 1.], f32::sqrt(2.));
        expect_intersect(tri2, ray_pos, ray_dir, [2., 0., 2.], f32::sqrt(2.));
    }

    #[test]
    fn non_intersection_1() {
        let ray_pos = Vec3::from_xyz(0., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 1., -1.);

        let tri = Tri { v: [Vec3::from_xyz(-1., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: None };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }

    #[test]
    fn non_intersection_2() {
        let ray_pos = Vec3::from_xyz(5., 0., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(-10., -1., 1.), Vec3::from_xyz(1., -1., 1.), Vec3::from_xyz(0., 1., 1.)], shader: None };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }

    #[test]
    fn non_intersection_3() {
        let ray_pos = Vec3::from_xyz(1., -1., 0.);
        let ray_dir = Vec3::from_xyz(0., 0., 1.);

        let tri = Tri { v: [Vec3::from_xyz(0.5, 0.5, 1.), Vec3::from_xyz(-0.5, 0.5, 1.), Vec3::from_xyz(0., -0.5, 1.)], shader: None };

        expect_no_intersect(tri, ray_pos, ray_dir);
    }
}

struct Sphere {
    center: Vec3, // typical notation in math is (h, k, j)
    radius: f32,
}

/// Linearly maps a value from a source range to a target range
fn map_range(s: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
    return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

/// h: [0, 360)
/// s: [0, 1]
/// v: [0, 1]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgbf32 {
    assert!(h >= 0. && h < 360.);
    assert!(s >= 0. && s <= 1.);
    assert!(v >= 0. && v <= 1.);

    // calculate chroma
    let c = v * s;

    // calculate secondary color component
    let x = c * (1. - f32::abs((h / 60.) % 2.) - 1.);

    // calculate match value
    let m = v - c;

    // r', g', b'
    let p: Vec3;

    if (h < 60.) {
        p = Vec3::from_xyz(c, x, 0.);
    } else if (h < 120.) {
        p = Vec3::from_xyz(x, c, 0.);
    } else if (h < 180.) {
        p = Vec3::from_xyz(0., c, x);
    } else if (h < 240.) {
        p = Vec3::from_xyz(0., x, c);
    } else if (h < 300.) {
        p = Vec3::from_xyz(x, 0., c);
    } else {
        p = Vec3::from_xyz(c, 0., x);
    }

    return p.add(Vec3::from_xyz(m, m, m));
}

type GeoId = usize;

struct GeosHandle {
    val: usize,
}

fn trongle() -> Geo {
    Geo::Tri(Tri {
            v: [
                Vec3::from_xyz(0.5, -0.5, 1.),
                Vec3::from_xyz(-0.5, -0.5, 1.),
                Vec3::from_xyz(0., 0.5, 1.)
            ],
            shader: Some(TriShader::PerVertex(
                TriPerVertexShader { colors: [
                    [ 1., 0., 0., 0.5 ],
                    [ 0., 1., 0., 0.5 ],
                    [ 0., 0., 1., 0.5 ]
                ]}
            ))
        },
    )
}

/// Square with side length `len`, centered on (0, 0, 0), normal facing (0, 0, 1).
fn square(len: f32, shader: TriShader) -> [Geo; 2] {
    let half_len = len;

    let t1 = Tri {
        v: [
            Vec3::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz(-half_len / 2.,  half_len / 2., 0.),
            Vec3::from_xyz( half_len / 2.,  half_len / 2., 0.)
        ],
        shader: Some(shader.clone()),
    };

    let t2 = Tri {
        v: [
            Vec3::from_xyz( half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3::from_xyz( half_len / 2.,  half_len / 2., 0.),
        ],
        shader: Some(shader.clone()),
    };

    let g1 = Geo::Tri(t1);
    let g2 = Geo::Tri(t2);

    return [g1, g2];
}
    
/// Cube with side length `len`, centered on (0, 0, 0).
fn cube(len: f32) -> [Geo; 12] {
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

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
fn random_jitter(prng: &mut StdRng) -> (f32, f32) {
    // prng.next_u64() returns integer with uniform distribution in the entire u64 domain
    let next = prng.next_u64();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    let x_jitter_int = next & 0xFFFFFFFF;
    let y_jitter_int = (next >> 32) & 0xFFFFFFFF;

    return (
        (x_jitter_int as f32) / (0xFFFFFFFFu32 as f32) - 0.5,
        (y_jitter_int as f32) / (0xFFFFFFFFu32 as f32) - 0.5,
    )
}

// 3D vector that is evenly distributed on the unit sphere
fn random_unit_vector(prng: &mut StdRng) -> Vec3 {
    loop {
        // prng.next_u64() returns integer with uniform distribution in the entire u64 domain
        let next = prng.next_u64();

        // split it in three to get three 21-bit vars where each bit is I.I.D.
        let xyz_int: [u64; 3] = [next & 0x1FFFFF,
                                (next >> 21) & 0x1FFFFF,
                                (next >> 42) & 0x1FFFFF];

        let mut xyz_float = Vec3::from_xyz(xyz_int[0] as f32,
                                            xyz_int[1] as f32,
                                            xyz_int[2] as f32);

        xyz_float = xyz_float
            .div_scalar(0x1FFFFF as f32)
            .sub(Vec3::from_xyz(-0.5, -0.5, -0.5))
            .mul_scalar(2.0);

        let xyz_magn = xyz_float.magn();
        
        if (xyz_magn <= 1.0 && xyz_magn > 0.01) {
            return xyz_float.normalize();
        }
    }
}

// return is random value in [0, 1]
fn random_alpha(prng: &mut StdRng) -> f32 {
    // prng.next_u64() returns integer with uniform distribution in the entire u64 domain
    let next = prng.next_u64();

    // split it in half to get two 32-bit vars where each bit is I.I.D.
    let int = next & 0xFFFFFFFF;

    return (int as f32) / (0xFFFFFFFFu32 as f32);
}

struct Raytracer {
    main_prng: rand::rngs::StdRng,
    dither_prng: rand::rngs::StdRng,
    geos: Vec<Geo>,
    handleToGeoIdsArrayList: Vec<Vec<GeoId>>,
    accumulator_buf: [[Rgbf32; WINDOW_HEIGHT]; WINDOW_WIDTH],
    accumulate_steps: u32,
}

impl Raytracer {
    fn new(&mut self) -> Self {
        // try add_trongle();

        let mut raytracer = Self {
            // start the PRNGs from 0 seed always
            main_prng: rand::rngs::StdRng::seed_from_u64(0),
            dither_prng: rand::rngs::StdRng::seed_from_u64(0),
            geos: Vec::new(),
            handleToGeoIdsArrayList: Vec::new(),
            accumulator_buf: [[Rgbf32::zero(); WINDOW_HEIGHT]; WINDOW_WIDTH],
            accumulate_steps: 0,
        };

        let mut light = square(1., TriShader::Material(MaterialShader::new_emissive(1., 1., 1., 4.)));
        transform_in_place(&mut light, Mat33::rotation(90., 0., 0.).translate(0., 0.999, 1.));
        raytracer.add_geos(&light);

        // *CORNELL BOX MENTIONED*

        let mut left_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 0., 0.)));
        transform_in_place(&mut left_wall, Mat33::rotation(  0.,  90.,  0.).translate( 1., 0., 1.25));
        raytracer.add_geos(&left_wall);
        
        let mut right_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(0., 1., 0.)));
        transform_in_place(&mut right_wall, Mat33::rotation(  0., -90.,  0.).translate(-1., 0., 1.25));
        raytracer.add_geos(&right_wall);
        
        let mut roof = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));
        transform_in_place(&mut roof, Mat33::rotation(  90.,  0.,  0.).translate(0.,  1., 1.25));
        raytracer.add_geos(&roof);
        
        let mut floor = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));
        transform_in_place(&mut floor, Mat33::rotation( -90.,  0.,  0.).translate(0., -1., 1.25));
        raytracer.add_geos(&floor);

        // try add_quad(Vec3::from_xyz(0, 0.999, 1), 1, 0, 0, 90, .{.material = MaterialShader::new_emissive(0, 0, 1, 1)}); // light

        let mut rear_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));    // rear wall
        transform_in_place(&mut rear_wall, Mat33::rotation(  0.,  0.,  0.).translate(0.,  0., 2.25));
        raytracer.add_geos(&rear_wall);

        let mut cube = cube(0.5);
        transform_in_place(&mut cube, Mat33::rotation(0., -70., 0.).translate(-0.25, -0.75, 1.25));
        raytracer.add_geos(&cube);

        raytracer
    }

    fn add_geos(&mut self, geos_to_add: &[Geo]) -> GeosHandle {
        let mut geo_indices = Vec::with_capacity(geos_to_add.len());
        for (i, _) in geos_to_add.iter().enumerate() { 
            geo_indices.push(self.geos.len() + i); 
        }

        for geo_to_add in geos_to_add {
            self.geos.push((*geo_to_add).clone());
        }

        let handle_to_return = self.handleToGeoIdsArrayList.len();
        self.handleToGeoIdsArrayList.push(geo_indices);

        return GeosHandle { val: handle_to_return };
    }

    fn update_and_dither_output_buffer(&mut self, buf: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT]) {
        self.dither_prng = rand::rngs::StdRng::seed_from_u64(0);

        for (row, row_pixels) in buf.iter_mut().enumerate() {
            for (col, px) in row_pixels.iter_mut().enumerate() {
                // prng.next_u64() returns integer with uniform distribution in the entire u64 domain
                let next = self.dither_prng.next_u64();

                // get bottom half to get a 32-bit var where each bit is I.I.D.
                let int = next & 0xFFFFFFFF;
                let uniform_rect_dist =
                    (int as f32) / (0xFFFFFFFFu32 as f32) - 0.5 * (1. / DITHER_DIVISOR as f32);

                let mut dithered = self.accumulator_buf[row][col].add(Rgbf32 {val: [
                    uniform_rect_dist,
                    uniform_rect_dist,
                    uniform_rect_dist,
                ]});

                *px = dithered.to_argb8888(1.);
            }
        }
    }

    pub fn zaytrace_image(&mut self, buf: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT]) -> u32 {
        if (self.accumulate_steps < MAX_ACCUMULATE_STEPS) {
            // **First, trace one ray per pixel, and then update the accumulator
            // buffer with the cumulative average.**
            for (row, row_pixels) in buf.iter_mut().enumerate() {
                // map row to normalized y coordinates
                //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
                //      and z points into the screen, away from the viewer.)

                for (col, _) in row_pixels.iter_mut().enumerate() {
                    let jitter = random_jitter(&mut self.main_prng);
                    let row_jittered = (row as f32) + jitter.0;
                    let col_jittered = (col as f32) + jitter.1;

                    let norm_y = map_range(row_jittered, 0., WINDOW_HEIGHT as f32, -1., 1.);
                    let norm_x = map_range(col_jittered, 0., WINDOW_WIDTH as f32, -1., 1.);

                    // perform antialiasing accumulate
                    // https://en.wikipedia.org/wiki/Moving_average#Cumulative_average
                    let new_px = self.zaytrace_image_inner(norm_x, norm_y);
                    let acc_px = &mut self.accumulator_buf[row][col];
                    *acc_px = new_px.add(acc_px.mul_scalar(self.accumulate_steps as f32))
                        .div((self.accumulate_steps + 1) as f32);
                }
            }

            // **Then, update the output buffer while dithering it.**
            self.update_and_dither_output_buffer(buf);

            self.accumulate_steps += 1;
        }

        return self.accumulate_steps;
    }

    // `norm_x` and `norm_y` are Vulkan-style normalized
    // device coordinates, where the top left of the screen
    // is (-1, -1) and the bottom right is (1, 1).
    fn zaytrace_image_inner(&mut self, norm_x: f32, flipped_norm_y: f32) -> Rgbf32 {
        // Flip y to make y up
        let norm_y = flipped_norm_y * -1.;

        // **Perspective Projection**
        const FOV_HORIZONTAL_RAD: f32 = deg_to_rad(35.);

        // Given a projection plane 1 unit ahead of the camera,
        // how far away from the center of the plane are the
        // left/right (x_outer_bound) and top/bottom (y_outer_bound) edges?
        let x_outer_bound = f32::tan(FOV_HORIZONTAL_RAD);
        let y_outer_bound = x_outer_bound * ((WINDOW_HEIGHT as f32) / (WINDOW_WIDTH as f32));

        let x_dir_unnormalized = norm_x * x_outer_bound;
        let y_dir_unnormalized = norm_y * y_outer_bound;

        let ray_pos = Vec3::from_xyz(0., 0., -2.);
        let ray_dir = Vec3::from_xyz(x_dir_unnormalized, y_dir_unnormalized, 1.).normalize();

        return self.zaytrace_ray(None, ray_pos, ray_dir, 0);
    }

    // **Inner raytracing loop** :D
    const MAX_BOUNCES: u8 = 16;
    fn zaytrace_ray(&mut self, maybe_origin_geo_id: Option<GeoId>, ray_pos: Vec3, ray_dir: Vec3, bounces: u8) -> Rgbf32 {
        if (bounces > Self::MAX_BOUNCES) { return Rgbf32::from_rgb_f32(0., 0., 0.); }

        let mut closest_inter_t: f32 = 1000.0;
        let mut maybe_closest_inter: Option<Intersection> = None;
        let mut maybe_closest_inter_geo_id: Option<GeoId> = None;

        for (geo_id, geo) in self.geos.iter().enumerate() {
            if let Some(origin_geo_id) = maybe_origin_geo_id && origin_geo_id == geo_id { continue };

            match (geo) {
                Geo::Tri(tri) => {
                    if let Some((inter, tri_inter)) = tri.intersect(ray_pos, ray_dir) {
                        if (inter.t < closest_inter_t) {
                            // TODO: alpha only applies to direct rays so far. add separate indirect/direct bounce settings
                            if (bounces == 0) {
                                let shader = tri.shader.as_ref().unwrap();
                                match shader {
                                    PerVertex(per_vertex) => {
                                        let w = tri_inter.w;
                                        let v = tri_inter.v;
                                        let u = tri_inter.u;

                                        let colors = per_vertex.colors;
                                        let a = w * colors[0][3] + v * colors[1][3] + u * colors[2][3];

                                        if (a > random_alpha(&mut self.main_prng)) {
                                            continue;
                                        }
                                    },
                                    _ => unimplemented!()
                                }
                            }

                            closest_inter_t = inter.t;
                            maybe_closest_inter = Some(inter);
                            maybe_closest_inter_geo_id = Some(geo_id);
                        }
                    }
                },
                _ => unimplemented!()
            }
        }

        if let Some(closest_inter) = maybe_closest_inter {
            let geo = closest_inter.specific;
            match geo {
                IntersectionGeo::Tri(tri_inter) => {
                    let tri = tri_inter.tri;
                    let shader = tri.shader.as_ref().unwrap();
                    match shader {
                        PerVertex(per_vertex) => {
                            // currently this is emissive
                            let w = tri_inter.w;
                            let v = tri_inter.v;
                            let u = tri_inter.u;

                            let colors = per_vertex.colors;
                            let r = w * colors[0][0] + v * colors[1][0] + u * colors[2][0];
                            let g = w * colors[0][1] + v * colors[1][1] + u * colors[2][1];
                            let b = w * colors[0][2] + v * colors[1][2] + u * colors[2][2];

                            // make it much brighter if it's indirect bounce
                            if (bounces > 0) {
                                return Rgbf32::from_rgb_f32(r, g, b).mul_scalar(10.);
                            } else {
                                return Rgbf32::from_rgb_f32(r, g, b);
                            }
                        },
                        Material(material) => {
                            if (material.emission >= 1.) {
                                // Do not bounce if fully emissive
                                if (bounces > 0) {
                                    return material.color.mul_scalar(material.emission);
                                } else {
                                    // To avoid color clipping, don't multiply direct rays by more than 1
                                    return material.color;
                                }
                            } else {
                                let mut n = closest_inter.n;
                                if (n.dot(ray_dir) > 0.) {
                                    // If normal is facing the wrong direction, flip it
                                    n = n.mul_scalar(-1.);
                                }

                                // **Find orthonormal basis**
                                // Building an Orthonormal Basis, Revisited - Pixar
                                // https://jcgt.org/published/0006/01/01/
                                // const x = n.val[0];
                                // const y = n.val[1];
                                // const z = n.val[2];
                                // const sign = std.math.copysign(@as(f32, 1), z); // std.math.copysign is not the same as std.math.sign
                                // const a = -1.0 / (sign + z);
                                // const b = x * y * a;
                                // const b1 = Vec3{ .val = .{ 1 + sign * x * x * a, sign * b, -sign * x } };
                                // const b2 = Vec3{ .val = .{ b, sign + y * y * a, -y } };
                                // now <n, b1, b2> is an orthonormal basis

                                // get a random 2D vector within the hemisphere that will bounce away from the surface
                                let mut bounce_dir = random_unit_vector(&mut self.main_prng);
                                if (n.dot(bounce_dir) < 0.) {
                                    bounce_dir = bounce_dir.mul_scalar(-1.);
                                }

                                // cos(θ) = a ⋅ b / |a||b|, but since bounce_dir and n are unit vectors, no need to divide by |a||b|
                                // const lambertian_scalar = n.dot(bounce_dir);

                                // assert(bounce_dir.dot(n) > 0);
                                return material.color
                                    .mul_elementwise(self.zaytrace_ray(maybe_closest_inter_geo_id, closest_inter.p, bounce_dir, bounces + 1));
                                    // .mul_scalar(lambertian_scalar);
                            }
                        },
                    }
                },
            }
        }

        return Rgbf32::from_rgb_f32(0., 0., 0.);
    }

    fn consolidate_handles(&mut self, handles: &[GeosHandle]) -> GeosHandle {
        let mut num_ids: usize = 0;
        for handle in handles { num_ids += self.resolve_handle(handle).len(); }
        let mut consolidated_ids = Vec::with_capacity(num_ids);

        let mut i: usize = 0;
        for handle in handles {
            for geo_id in self.resolve_handle(handle) {
                consolidated_ids[i] = *geo_id; 
                i += 1;
            }
        }

        self.handleToGeoIdsArrayList.push(consolidated_ids);

        return GeosHandle { val: self.handleToGeoIdsArrayList.len() - 1 };
    }    

    fn resolve_handle(&mut self, handle: &GeosHandle) -> &Vec<GeoId> {
        return &self.handleToGeoIdsArrayList[handle.val];
    }
}

pub fn rainbow(buf: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT]) {
    for (row, row_pixels) in buf.iter_mut().enumerate() {
        // map row to normalized y coordinates
        //     (Vulkan style, [-1, -1] xy is the top left corner of the screen)
        let y = map_range((row as f32), 0., WINDOW_HEIGHT as f32, -1.0, 1.0);
        for (col, px) in row_pixels.iter_mut().enumerate() {
            let x = map_range((col as f32), 0., WINDOW_WIDTH as f32, -1.0, 1.0);
            _ = x;

            let hue_deg = map_range(y, -1., 1., 0., 360.);
            *px = hsv_to_rgb(hue_deg, 1., 1.).to_argb8888(1.0);
        }
    }
}

const MAX_ACCUMULATE_STEPS: u32 = 1024;
const DITHER_DIVISOR: usize = 256;

const fn deg_to_rad(deg: f32) -> f32 {
    return deg * PI / 180.;
}

