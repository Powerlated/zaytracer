pub mod avx512;
pub mod rng_portable_simd;

use std::arch::x86_64::*;
use std::f32::consts::PI;
use std::ops::Div;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{f32x16, i32x16, simd_swizzle, u8x16, u32x16, u64x16};

use rand::rngs::Xoshiro128PlusPlus;
use rand::{Rng, SeedableRng};

use crate::raytracer::*;
use crate::raytracer::scalar::RaytracerScalar;
use crate::raytracer::scalar::geometry::{Geo, Intersection, IntersectionGeo, cube, square, transform_in_place};
use crate::raytracer::scalar::math::{Mat33, Bgra, Vec3, deg_to_rad, hsv_to_rgb, map_range};
use crate::raytracer::scalar::shader::TriShader::{Material, PerVertex};
use crate::raytracer::scalar::shader::{MaterialShader, TriShader};

type GeoId = usize;

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
fn random_jitter(prng: &mut Xoshiro128PlusPlus) -> (f32, f32) {
    let x_jitter_int = prng.next_u32() & 0xFFFFFFFF;
    let y_jitter_int = prng.next_u32() & 0xFFFFFFFF;

    return (
        (x_jitter_int as f32) / (0xFFFFFFFFu32 as f32) - 0.5,
        (y_jitter_int as f32) / (0xFFFFFFFFu32 as f32) - 0.5,
    )
}

// 3D vector that is evenly distributed on the unit sphere
fn random_unit_vector(prng: &mut Xoshiro128PlusPlus) -> Vec3 {
    loop {
        // get 64 bits of randomness
        let next = ((prng.next_u32() as u64) << 32) | (prng.next_u32() as u64);

        // split it in three to get three 21-bit vars where each bit is I.I.D.
        let xyz_int: [u64; 3] = [next & 0x1FFFFF,
                                (next >> 21) & 0x1FFFFF,
                                (next >> 42) & 0x1FFFFF];

        let mut xyz_float = Vec3::from_xyz(xyz_int[0] as f32,
                                            xyz_int[1] as f32,
                                            xyz_int[2] as f32);

        xyz_float = xyz_float
            .div_scalar(0x1FFFFF as f32)
            .sub(Vec3::from_xyz(0.5, 0.5, 0.5))
            .mul_scalar(2.0);

        let xyz_magn = xyz_float.magn();
        
        if xyz_magn <= 1.0 && xyz_magn > 0.01  {
            return xyz_float.normalize();
        }
    }
}

// return is random value in [0, 1]
fn random_alpha(prng: &mut Xoshiro128PlusPlus) -> f32 {
    let int = prng.next_u32() & 0xFFFFFFFF;
    return (int as f32) / (0xFFFFFFFFu32 as f32);
}

pub struct RaytracerSimd {
    main_prng: Xoshiro128PlusPlus,
    geos: Vec<Geo>,
    accumulator_buf: Box<[Bgra]>,
    accumulate_steps: u32,
}

impl RaytracerSimd {
    pub fn from_scalar(raytracer_scalar: RaytracerScalar) -> Self {
        Self {
            // start the PRNGs from 0 seed always
            main_prng: Xoshiro128PlusPlus::seed_from_u64(0),
            geos: raytracer_scalar.geos,
            accumulator_buf: vec![Bgra::zero(); HEIGHT * WIDTH as usize].into_boxed_slice(),
            accumulate_steps: 0,
        }
    }

    fn update_and_dither_output_buffer(&mut self, buf: &mut RaytracerBuf) {
        let mut dither_prng = rand::rngs::Xoshiro128PlusPlus::seed_from_u64(0);

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                // prng.next_u64() returns integer with uniform distribution in the entire u64 domain
                let next = dither_prng.next_u64();

                // get bottom half to get a 32-bit var where each bit is I.I.D.
                let int = next & 0xFFFFFFFF;
                let uniform_rect_dist = ((int as f32) / (0xFFFFFFFFu32 as f32) - 0.5)
                                        * (1. / (DITHER_DIVISOR as f32));

                let scaled = self.accumulator_buf[row * WIDTH + col]
                            .div_scalar(self.accumulate_steps as f32);
                let dithered = scaled.add(Bgra {val: [
                    uniform_rect_dist,
                    uniform_rect_dist,
                    uniform_rect_dist,
                    0.0,
                ]});

                let buf_offset = row * STRIDE + col * BYTES_PER_PIXEL;
                let px: &mut [u8; 4] = (&mut buf[buf_offset..buf_offset+BYTES_PER_PIXEL])
                                        .try_into().unwrap();

                *px = dithered.to_argb8888();
            }
        }   
    }

    pub fn raytrace_step(&mut self, buf: &mut RaytracerBuf) -> u32 {
        if self.accumulate_steps < MAX_ACCUMULATE_STEPS  {
            for row in 0..HEIGHT {
                // map row to normalized y coordinates
                //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
                //      and z points into the screen, away from the viewer.)

                for col in 0..WIDTH {
                    // **First, trace one ray per pixel, and then update the accumulator
                    // buffer with the cumulative average.**

                    let jitter = random_jitter(&mut self.main_prng);
                    let row_jittered = (row as f32) + jitter.0;
                    let col_jittered = (col as f32) + jitter.1;

                    let norm_y = map_range(row_jittered, 0., HEIGHT as f32, -1., 1.);
                    let norm_x = map_range(col_jittered, 0., WIDTH as f32, -1., 1.);

                    // perform antialiasing accumulate
                    let new_px = self.zaytrace_image_inner(norm_x, norm_y);
                    let acc_px = &mut self.accumulator_buf[row * WIDTH + col];
                    *acc_px = (*acc_px).add(new_px);
                }
            }

            // **Then, update the output buffer while dithering it.**
            self.accumulate_steps += 1;
            self.update_and_dither_output_buffer(buf);

        }

        return self.accumulate_steps;
    }

    // `norm_x` and `norm_y` are Vulkan-style normalized
    // device coordinates, where the top left of the screen
    // is (-1, -1) and the bottom right is (1, 1).
    fn zaytrace_image_inner(&mut self, norm_x: f32, flipped_norm_y: f32) -> Bgra {
        // Flip y to make y up
        let norm_y = flipped_norm_y * -1.;

        // **Perspective Projection**
        const FOV_HORIZONTAL_RAD: f32 = deg_to_rad(35.);

        // Given a projection plane 1 unit ahead of the camera,
        // how far away from the center of the plane are the
        // left/right (x_outer_bound) and top/bottom (y_outer_bound) edges?
        let x_outer_bound = f32::tan(FOV_HORIZONTAL_RAD);
        let y_outer_bound = x_outer_bound * ((HEIGHT as f32) / (WIDTH as f32));

        let x_dir_unnormalized = norm_x * x_outer_bound;
        let y_dir_unnormalized = norm_y * y_outer_bound;

        let ray_pos = Vec3::from_xyz(0., 0., -2.);
        let ray_dir = Vec3::from_xyz(x_dir_unnormalized, y_dir_unnormalized, 1.).normalize();

        return self.zaytrace_ray(None, ray_pos, ray_dir, 0);
    }

    // **Inner raytracing loop** :D
    const MAX_BOUNCES: u8 = 16;
    fn zaytrace_ray(&mut self, maybe_origin_geo_id: Option<GeoId>, ray_pos: Vec3, ray_dir: Vec3, bounces: u8) -> Bgra {
        if bounces > Self::MAX_BOUNCES  { return Bgra::from_rgb_f32(0., 0., 0.); }

        let mut closest_inter_t: f32 = 1000.0;
        let mut maybe_closest_inter: Option<Intersection> = None;
        let mut maybe_closest_inter_geo_id: Option<GeoId> = None;

        for (geo_id, geo) in self.geos.iter().enumerate() {
            if let Some(origin_geo_id) = maybe_origin_geo_id && origin_geo_id == geo_id { continue };

            match geo  {
                Geo::Tri(tri) => {
                    if let Some((inter, tri_inter)) = tri.intersect(ray_pos, ray_dir) {
                        if inter.t < closest_inter_t  {
                            // TODO: alpha only applies to direct rays so far. add separate indirect/direct bounce settings
                            if bounces == 0  {
                                match &tri.shader {
                                    PerVertex(per_vertex) => {
                                        let w = tri_inter.w;
                                        let v = tri_inter.v;
                                        let u = tri_inter.u;

                                        let colors = per_vertex.colors;
                                        let a = w * colors[0][3] + v * colors[1][3] + u * colors[2][3];

                                        if a > random_alpha(&mut self.main_prng)  {
                                            continue;
                                        }
                                    },
                                    _ => {}
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
                    match &tri.shader {
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
                            if bounces > 0  {
                                return Bgra::from_rgb_f32(r, g, b).mul_scalar(10.);
                            } else {
                                return Bgra::from_rgb_f32(r, g, b);
                            }
                        },
                        Material(material) => {
                            if material.emission >= 1.  {
                                // Do not bounce if fully emissive
                                if bounces > 0  {
                                    return material.color.mul_scalar(material.emission);
                                } else {
                                    // To avoid color clipping, don't multiply direct rays by more than 1
                                    return material.color;
                                }
                            } else {
                                let mut n = closest_inter.n;
                                if n.dot(ray_dir) > 0.  {
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

                                // **Lambertian diffuse bounce**
                                // get a random point on the unit sphere around the hit location
                                let point_on_sphere = closest_inter.p.add(random_unit_vector(&mut self.main_prng));
                                // shift that point 1 unit away from the hit point along the normal vector
                                let point_on_sphere_shifted = point_on_sphere.add(n);
                                let lambertian_ray_dir = point_on_sphere_shifted.sub(closest_inter.p).normalize();
                                // lambertian_ray_dir now should have a lambertian distribution

                                // assert(bounce_dir.dot(n) > 0);
                                return material.color
                                    .mul_elementwise(self.zaytrace_ray(maybe_closest_inter_geo_id, closest_inter.p, lambertian_ray_dir, bounces + 1));
                                    // .mul_scalar(lambertian_scalar);
                            }
                        },
                    }
                },
            }
        }

        return Bgra::from_rgb_f32(0., 0., 0.);
    }
}

pub const MAX_ACCUMULATE_STEPS: u32 = 128;
pub const DITHER_DIVISOR: usize = 256;