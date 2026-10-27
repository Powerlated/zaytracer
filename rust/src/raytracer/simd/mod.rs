pub mod avx512;
pub mod geometry;
pub mod rng_portable_simd;

use std::arch::x86_64::*;
use std::f32::consts::PI;
use std::ops::Div;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{f32x16, i32x16, simd_swizzle, u8x16, u32x16, u64x4, u64x16};

use bytemuck::offset_of;
use rand::rngs::Xoshiro128PlusPlus;
use rand::{Rng, SeedableRng};
use wgpu::naga::VectorSize::Tri;

use crate::raytracer::math::{
    Bgra, Bgraf32, Mat33f32, Vec3, Vec3f32, deg_to_rad, hsv_to_rgb, map_range,
};
use crate::raytracer::scalar::RaytracerScalar;
use crate::raytracer::scalar::geometry::{
    Geo, Intersection, IntersectionGeo, cube, square, transform_in_place,
};
use crate::raytracer::scalar::shader::TriShader::{Material, PerVertex};
use crate::raytracer::scalar::shader::{MaterialShader, TriShader};
use crate::raytracer::simd::avx512::*;
use crate::raytracer::simd::geometry::{TriIntersectionSimd, TriSimd};
use crate::raytracer::simd::rng_portable_simd::Xoshiro128PlusPlusSimd;
use crate::raytracer::*;

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
fn random_jitter(prng: &mut Xoshiro128PlusPlusSimd<16>) -> (F32x16, F32x16) {
    let x_jitter_int = prng.next_u32();
    let y_jitter_int = prng.next_u32();

    return (
        F32x16::from(x_jitter_int) / (0xFFFFFFFFu32 as f32) - 0.5,
        F32x16::from(y_jitter_int) / (0xFFFFFFFFu32 as f32) - 0.5,
    );
}

// 3D vector that is evenly distributed on the unit sphere
fn random_unit_vector(prng: &mut Xoshiro128PlusPlus) -> Vec3f32 {
    loop {
        // get 64 bits of randomness
        let next = ((prng.next_u32() as u64) << 32) | (prng.next_u32() as u64);

        // split it in three to get three 21-bit vars where each bit is I.I.D.
        let xyz_int: [u64; 3] = [
            next & 0x1FFFFF,
            (next >> 21) & 0x1FFFFF,
            (next >> 42) & 0x1FFFFF,
        ];

        let mut xyz_float =
            Vec3f32::from_xyz(xyz_int[0] as f32, xyz_int[1] as f32, xyz_int[2] as f32);

        xyz_float = xyz_float
            .div_scalar(0x1FFFFF as f32)
            .sub(Vec3f32::from_xyz(0.5, 0.5, 0.5))
            .mul_scalar(2.0);

        let xyz_magn = xyz_float.magn();

        if xyz_magn <= 1.0 && xyz_magn > 0.01 {
            return xyz_float.normalize();
        }
    }
}

// return is random value in [0, 1]
fn random_alpha(prng: &mut Xoshiro128PlusPlus) -> f32 {
    let int = prng.next_u32() & 0xFFFFFFFF;
    return (int as f32) / (0xFFFFFFFFu32 as f32);
}

const ACCUMULATOR_BUF_LEN: usize = HEIGHT * WIDTH as usize / 16;

pub struct RaytracerSimd {
    main_prng: Xoshiro128PlusPlusSimd<16>,
    scalar_prng: Xoshiro128PlusPlus,
    geos: Vec<TriSimd>,
    shaders: Vec<MaterialShader>,
    accumulator_buf: Box<[Bgra<F32x16>; ACCUMULATOR_BUF_LEN]>,
    accumulate_steps: u32,
}

impl RaytracerSimd {
    pub fn from_scalar(raytracer_scalar: RaytracerScalar) -> Self {
        let mut geos = Vec::new();
        let mut shaders = Vec::new();

        for geo in raytracer_scalar.geos {
            match geo {
                Geo::Tri(tri) => {
                    geos.push(TriSimd { v: tri.v });
                    shaders.push(match tri.shader {
                        Material(material) => material,
                        _ => unimplemented!(),
                    });
                }
                _ => unimplemented!(),
            }
        }

        Self {
            // start the PRNGs from consistent seed always
            main_prng: Xoshiro128PlusPlusSimd::<16>::seed_from_u64(u64x16::from(
                std::array::from_fn::<u64, 16, _>(|i| i as u64),
            )),
            scalar_prng: Xoshiro128PlusPlus::seed_from_u64(0),
            geos,
            shaders,
            accumulator_buf: vec![Bgra::<F32x16>::zero(); ACCUMULATOR_BUF_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            accumulate_steps: 0,
        }
    }

    fn update_and_dither_output_buffer(&mut self, buf: &mut RaytracerBuf) {
        let mut dither_prng =
            Xoshiro128PlusPlusSimd::<16>::seed_from_u64(u64x16::from(std::array::from_fn::<
                u64,
                16,
                _,
            >(|i| i as u64)));

        // 16 pixels
        let out_buf: &mut [[u8; 16 * 4]] = bytemuck::cast_slice_mut(buf);
        // An Rgbf32::<F32x16> is 16 pixels

        // 16 pixels per iteration
        for (acc, out) in self.accumulator_buf.iter().zip(out_buf.iter_mut()) {
            let int = dither_prng.next_u32();

            let uniform_rect_dist =
                (F32x16::from(int) / (0xFFFFFFFFu32 as f32) - 0.5) * (1. / (DITHER_DIVISOR as f32));

            let inv_accumulate_steps: F32x16 = (1.0 / self.accumulate_steps as f32).into();
            let dithered_f32_b = acc.val[0] * inv_accumulate_steps + uniform_rect_dist;
            let dithered_f32_g = acc.val[1] * inv_accumulate_steps + uniform_rect_dist;
            let dithered_f32_r = acc.val[2] * inv_accumulate_steps + uniform_rect_dist;
            let scaled_f32_a = acc.val[3] * inv_accumulate_steps;

            // // Construct the first 4 pixels in f32
            // unsafe {
            //     // Construct BG
            //     // let permute = _mm512_permutex2var_ps;
            //     // idx[i+3:i] is the index into the source vectors
            //     // idx[i+4] should be 1 to select b
            //     let indices: [u8; 16] = [
            //         0,
            //         0,
            //         0b00000,
            //         0b10000,
            //         0,
            //         0,
            //         0b00001,
            //         0b10001,
            //         0,
            //         0,
            //         0b00010,
            //         0b10010,
            //         0,
            //         0,
            //         0b00011,
            //         0b10011,
            //     ];
            //     let indices = _mm_loadu_epi8(indices.as_ptr() as *const i8);
            //     let indices = _mm512_cvtepi8_epi32(indices);
            //     let mask: u16 = 0b1100110011001100;
            //     _mm512_mask2_permutex2var_ps(dithered_f32_b.0, indices, mask, dithered_f32_g.0);
            // }

            let dithered_f32_b_arr: [f32; 16] = dithered_f32_b.into();
            let dithered_f32_g_arr: [f32; 16] = dithered_f32_g.into();
            let dithered_f32_r_arr: [f32; 16] = dithered_f32_r.into();

            // TODO: Vectorize this
            for i in 0..16 {
                let out_pixel = &mut out[i * 4..i * 4 + 4];
                out_pixel[0] = ((dithered_f32_b_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
                out_pixel[1] = ((dithered_f32_g_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
                out_pixel[2] = ((dithered_f32_r_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
                out_pixel[3] = 0xFF;
            }

            // let color_255: U8x16 = ((dithered_f32.clamp(0.0, 1.0) * 255.0) + 0.5).into();
        }
    }

    pub fn raytrace_step(&mut self, buf: &mut RaytracerBuf) -> u32 {
        if self.accumulate_steps < MAX_ACCUMULATE_STEPS {
            for row in 0..HEIGHT {
                // map row to normalized y coordinates
                //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
                //      and z points into the screen, away from the viewer.)

                for col_stepped in (0..WIDTH).step_by(16) {
                    // **First, trace one ray per pixel, and then update the accumulator
                    // buffer with the cumulative average.**

                    let col_offs = F32x16::from(&std::array::from_fn::<_, 16, _>(|i| i as f32));
                    let col = F32x16::from(col_stepped as f32) + col_offs;

                    let jitter = random_jitter(&mut self.main_prng);

                    let row_jittered = F32x16::from(row as f32) + jitter.0;
                    let col_jittered = col + jitter.1;

                    let norm_y = map_range(row_jittered, 0.0, HEIGHT as f32 - 1.0, -1.0, 1.0);
                    let norm_x = map_range(col_jittered, 0.0, WIDTH as f32 - 1.0, -1.0, 1.0);

                    let new_pixels = self.zaytrace_image_inner(norm_x, norm_y);

                    // perform antialiasing accumulate
                    let acc_pixels = &mut self.accumulator_buf[(row * WIDTH + col_stepped) / 16];
                    *acc_pixels = (*acc_pixels).add(new_pixels);
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
    fn zaytrace_image_inner(&mut self, norm_x: F32x16, flipped_norm_y: F32x16) -> Bgra<F32x16> {
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

        let ray_pos = Vec3f32::from_xyz(0.0, 0.0, -2.0);
        let ray_dir = Vec3::<F32x16>::from_xyz(x_dir_unnormalized, y_dir_unnormalized, 1.0.into())
            .normalize();

        return self.zaytrace_ray(ray_pos, ray_dir, 0);
    }

    // **Inner raytracing loop** :D
    const MAX_BOUNCES: u8 = 16;
    fn zaytrace_ray(
        &mut self,
        ray_pos: Vec3f32,
        ray_dir: Vec3<F32x16>,
        bounces: u8,
    ) -> Bgra<F32x16> {
        // TODO: Bounce limit
        // if bounces > Self::MAX_BOUNCES {
        //     return Bgraf32::from_rgb_f32(0., 0., 0.);
        // }

        unsafe {
            let mut closest_inter_t = F32x16::splat(1000.0);
            let mut hitmask: u16 = 0;
            let mut hit_ids = U32x16::splat(0);

            for (tri_id, tri) in self.geos.iter().enumerate() {
                // TODO: Prevent rays from hitting their origin triangle.
                // if let Some(origin_geo_id) = maybe_origin_geo_id
                //     && origin_geo_id == geo_id
                // {
                //     continue;
                // };

                let inter = tri.intersect(ray_pos, ray_dir);
                let mut writemask = inter.hitmask & inter.t.cmp_lt(closest_inter_t);

                // Prevent rays from hitting their origin triangle
                writemask &= !hit_ids.cmp_eq(tri_id as u32);

                closest_inter_t.0 = _mm512_mask_mov_ps(closest_inter_t.0, writemask, inter.t.0);
                hitmask |= writemask;
                hit_ids = U32x16::from(_mm512_mask_mov_epi32(hit_ids.0, writemask, _mm512_set1_epi32(tri_id as i32)));

                // TODO: alpha
            }

            // if let Some(closest_inter) = maybe_closest_inter {
            //     let geo = closest_inter.specific;
            //     match geo {
            //         IntersectionGeo::Tri(tri_inter) => {
            //             let tri = tri_inter.tri;
            //             match &tri.shader {
            //                 PerVertex(per_vertex) => {
            //                     // currently this is emissive
            //                     let w = tri_inter.w;
            //                     let v = tri_inter.v;
            //                     let u = tri_inter.u;
            //
            //                     let colors = per_vertex.colors;
            //                     let r = w * colors[0][0] + v * colors[1][0] + u * colors[2][0];
            //                     let g = w * colors[0][1] + v * colors[1][1] + u * colors[2][1];
            //                     let b = w * colors[0][2] + v * colors[1][2] + u * colors[2][2];
            //
            //                     // make it much brighter if it's indirect bounce
            //                     if bounces > 0 {
            //                         return Bgraf32::from_rgb_f32(r, g, b).mul_scalar(10.);
            //                     } else {
            //                         return Bgraf32::from_rgb_f32(r, g, b);
            //                     }
            //                 }
            //                 Material(material) => {
            //                     if material.emission >= 1. {
            //                         // Do not bounce if fully emissive
            //                         if bounces > 0 {
            //                             return material.color.mul_scalar(material.emission);
            //                         } else {
            //                             // To avoid color clipping, don't multiply direct rays by more than 1
            //                             return material.color;
            //                         }
            //                     } else {
            //                         let mut n = closest_inter.n;
            //                         if n.dot(ray_dir) > 0. {
            //                             // If normal is facing the wrong direction, flip it
            //                             n = n.mul_scalar(-1.);
            //                         }
            //
            //                         let point_on_sphere =
            //                             closest_inter.p.add(random_unit_vector(&mut self.scalar_prng));
            //                         let point_on_sphere_shifted = point_on_sphere.add(n);
            //                         let lambertian_ray_dir =
            //                             point_on_sphere_shifted.sub(closest_inter.p).normalize();
            //                         // lambertian_ray_dir now should have a lambertian distribution
            //
            //                         return material.color.mul_elementwise(self.zaytrace_ray(
            //                             maybe_closest_inter_geo_id,
            //                             closest_inter.p,
            //                             lambertian_ray_dir,
            //                             bounces + 1,
            //                         ));
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }

            let size = U32x16::splat(size_of::<MaterialShader>() as u32);
            
            let mut ret = Bgra::<F32x16>::from_rgb_f32(0., 0., 0.);
            
            // blue, green, and then red
            for i in 0..3 {
                let bgraf32_offset = offset_of!(MaterialShader, color); // in bytes
                let channel_offset = U32x16::splat((bgraf32_offset + i*4) as u32); // in bytes

                let color_offsets = U32x16::from(_mm512_mullo_epi32(hit_ids.0, size.0)) + channel_offset;
                ret.val[i] = F32x16::from(_mm512_mask_i32gather_ps::<1>(
                    ret.val[i].0,
                    hitmask,
                    color_offsets.0,
                    self.shaders.as_ptr() as *const f32,
                ));
            }

            return ret;
        }
    }
}
