pub mod avx512;
pub mod geometry;
pub mod rng_portable_simd;

use std::arch::x86_64::*;
use std::f32::consts::PI;
use std::ops::Div;
use std::simd::{u32x16, u64x16};

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

struct RandomUnitVectorGenerator {
    prng: Xoshiro128PlusPlusSimd<16>,
    buf: [[f32; 32]; 3],
    buf_pos: usize,
}

impl RandomUnitVectorGenerator {
    fn new() -> Self {
        Self {
            prng: Xoshiro128PlusPlusSimd::<16>::seed_from_u64(u64x16::from(
                std::array::from_fn::<u64, 16, _>(|i| i as u64),
            )),
            buf: [[0.0; 32]; 3],
            buf_pos: 0,
        }
    }

    // 3D unit vector that is evenly distributed on the unit sphere
    fn next(&mut self) -> Vec3<F32x16> {
        loop {
            // get three random values - 96 bits of randomness per vector
            let xyz_int: [u32x16; 3] = [
                self.prng.next_u32(),
                self.prng.next_u32(),
                self.prng.next_u32(),
            ];

            let xyz_float = Vec3::<F32x16>::from_xyz(
                F32x16::from(xyz_int[0]),
                F32x16::from(xyz_int[1]),
                F32x16::from(xyz_int[2])
            );

            // map [0, 0xFFFFFFFF] to [-1, 1]
            let xyz_float = xyz_float
                .div_scalar(0xFFFFFFFFu32 as f32)
                .sub(Vec3::<F32x16>::from_xyz(0.5.into(), 0.5.into(), 0.5.into()))
                .mul_scalar(2.0);

            let xyz_magn = xyz_float.magn();
            // vectors with magnitude less than 1 and greater than 0.01 are valid
            let valid_mask = xyz_magn.cmp_le(1.0) & xyz_magn.cmp_gt(0.01);

            // after this, all vectors that satisfy the valid condition will be I.I.D. unit vectors
            let xyz_float = xyz_float.normalize();

            // write the valid unit vectors into the output buffer
            unsafe {
                _mm512_mask_compressstoreu_ps((&mut self.buf[0][self.buf_pos]) as *mut f32, valid_mask, xyz_float.val[0].0);
                _mm512_mask_compressstoreu_ps((&mut self.buf[1][self.buf_pos]) as *mut f32, valid_mask, xyz_float.val[1].0);
                _mm512_mask_compressstoreu_ps((&mut self.buf[2][self.buf_pos]) as *mut f32, valid_mask, xyz_float.val[2].0);
                self.buf_pos += valid_mask.count_ones() as usize;
                
                // we have enough valid unit vectors to return 16 unit vectors
                if self.buf_pos >= 16 {
                    self.buf_pos -= 16;
                    
                    return Vec3::<F32x16> {
                        val: [
                            F32x16::from_array(self.buf[0][self.buf_pos..self.buf_pos+16].try_into().unwrap()),
                            F32x16::from_array(self.buf[1][self.buf_pos..self.buf_pos+16].try_into().unwrap()),
                            F32x16::from_array(self.buf[2][self.buf_pos..self.buf_pos+16].try_into().unwrap())
                        ]
                    }
                }
            }
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
    random_unit_vector_gen: RandomUnitVectorGenerator,
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
            random_unit_vector_gen: RandomUnitVectorGenerator::new(),
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

        return self.zaytrace_ray(U32x16::from(0xFFFFFFFFu32), ray_pos.splat(), ray_dir, 0, 0xFFFF);
    }

    // **Inner raytracing loop** :D
    // Traces 16 rays in parallel
    fn zaytrace_ray(
        &mut self,
        origin_ids: U32x16,
        ray_pos: Vec3<F32x16>,
        ray_dir: Vec3<F32x16>,
        bounces: u8,
        mut active_mask: u16 // rays that are still 'alive' i.e. still bouncing around
    ) -> Bgra<F32x16> {
        // TODO: Bounce limit
        if bounces > MAX_BOUNCES {
            return Bgra::<F32x16>::from_rgb_f32(0.0, 0.0, 0.0);
        }

        unsafe {
            let mut hit_mask: u16 = 0; // rays, that in this current bounce, have hit a triangle
            let mut hit_ids = U32x16::splat(0xFFFFFFFF);
            
            let mut hit_t = F32x16::splat(1000.0);
            let mut hit_n = Vec3::<F32x16>::zero();
            let mut hit_pos = Vec3::<F32x16>::zero();

            assert!(self.geos.len() > 0);
            for (tri_id, tri) in self.geos.iter().enumerate() {
                let hit = tri.intersect(ray_pos, ray_dir);

                {
                    let mut write_mask = hit.hitmask & hit.t.cmp_lt(hit_t);
                    
                    // Prevent rays from hitting their origin triangle
                    write_mask &= origin_ids.cmp_neq(tri_id as u32);

                    // Update the hit_<x> variables for rays we've found a closer intersection for this iteration
                    hit_t.0 = _mm512_mask_mov_ps(hit_t.0, write_mask, hit.t.0);
                    hit_n.val[0].0 = _mm512_mask_mov_ps(hit_n.val[0].0, write_mask, _mm512_set1_ps(hit.n.val[0])); 
                    hit_n.val[1].0 = _mm512_mask_mov_ps(hit_n.val[1].0, write_mask, _mm512_set1_ps(hit.n.val[1])); 
                    hit_n.val[2].0 = _mm512_mask_mov_ps(hit_n.val[2].0, write_mask, _mm512_set1_ps(hit.n.val[2])); 
                    hit_pos.val[0].0 = _mm512_mask_mov_ps(hit_pos.val[0].0, write_mask, hit.p.val[0].0); 
                    hit_pos.val[1].0 = _mm512_mask_mov_ps(hit_pos.val[1].0, write_mask, hit.p.val[1].0); 
                    hit_pos.val[2].0 = _mm512_mask_mov_ps(hit_pos.val[2].0, write_mask, hit.p.val[2].0); 

                    hit_mask |= write_mask;
                    hit_ids = U32x16::from(_mm512_mask_mov_epi32(hit_ids.0, write_mask, _mm512_set1_epi32(tri_id as i32)));
                }

                // TODO: alpha
            }

            active_mask &= hit_mask;
            
            let mut material_color = Bgra::<F32x16>::from_rgb_f32(0., 0., 0.);
            let mut material_emission = F32x16::zero();
            
            // **Look up material for rays**
            {
                let material_size = U32x16::splat(size_of::<MaterialShader>() as u32);
                let material_index = U32x16::from(_mm512_mullo_epi32(hit_ids.0, material_size.0));
                let material_color_offset = offset_of!(MaterialShader, color); // in bytes
                let material_emission_offset = offset_of!(MaterialShader, emission); // in bytes
                // Look up color. Blue, green, and then red
                for i in 0..3 {
                    let channel_offset = U32x16::splat((material_color_offset + i*4) as u32); // in bytes
                    
                    let color_index = material_index + channel_offset;
                    material_color.val[i] = F32x16::from(_mm512_mask_i32gather_ps::<1>(
                        material_color.val[i].0,
                        hit_mask,
                        color_index.0,
                        self.shaders.as_ptr() as *const f32,
                    ));
                }
                
                let emission_index = material_index + U32x16::splat(material_emission_offset as u32);
                material_emission = F32x16::from(_mm512_mask_i32gather_ps::<1>(
                    material_emission.0,
                    hit_mask,
                    emission_index.0,
                    self.shaders.as_ptr() as *const f32,
                ));
            }

            let mut ret = Bgra::<F32x16>::from_rgb_f32(0.0, 0.0, 0.0);
            let emissive_mask = material_emission.cmp_ge(1.0);

            // **Handle emission**
            if emissive_mask != 0 {
                // Do not bounce if fully emissive
                active_mask &= !emissive_mask;

                if bounces > 0 {
                    for i in 0..3 {
                        ret.val[i] = F32x16::from(_mm512_mask_mov_ps(ret.val[i].0, emissive_mask, (material_color.val[i] * material_emission).0)) 
                    }
                } else {
                    // To avoid color clipping, don't apply gain to direct rays
                    for i in 0..3 {
                        ret.val[i] = F32x16::from(_mm512_mask_mov_ps(ret.val[i].0, emissive_mask, material_color.val[i].0)) 
                    }
                }
            }

            // **Handle diffuse**
            if emissive_mask != 0xFFFF {
                let mut n = hit_n;

                // If normal is facing the wrong direction, flip sign
                {
                    let normal_facing_wrong_direction_mask = n.dot(ray_dir).cmp_gt(0.0);
                    // Evil floating point bit level magic hacking??????
                    let sign_bit: __m512 = std::mem::transmute(_mm512_set1_epi32(0x80000000u32 as i32));
                    n.val[0].0 = _mm512_mask_xor_ps(n.val[0].0, normal_facing_wrong_direction_mask, n.val[0].0, sign_bit);
                    n.val[1].0 = _mm512_mask_xor_ps(n.val[1].0, normal_facing_wrong_direction_mask, n.val[1].0, sign_bit);
                    n.val[2].0 = _mm512_mask_xor_ps(n.val[2].0, normal_facing_wrong_direction_mask, n.val[2].0, sign_bit);
                }

                let point_on_sphere = hit_pos.add(self.random_unit_vector_gen.next());
                let point_on_sphere_shifted = point_on_sphere.add(n);
                let lambertian_ray_dir = point_on_sphere_shifted.sub(hit_pos).normalize();

                // Bounce if there are still rays active 
                if active_mask != 0 {
                    let bounced = self.zaytrace_ray(
                        hit_ids,
                        hit_pos,
                        lambertian_ray_dir,
                        bounces + 1,
                        active_mask
                    );

                    ret.val[0].0 = _mm512_mask_mul_ps(ret.val[0].0, active_mask, material_color.val[0].0, bounced.val[0].0);
                    ret.val[1].0 = _mm512_mask_mul_ps(ret.val[1].0, active_mask, material_color.val[1].0, bounced.val[1].0);
                    ret.val[2].0 = _mm512_mask_mul_ps(ret.val[2].0, active_mask, material_color.val[2].0, bounced.val[2].0);
                    // ret.val[3].0 = _mm512_mask_mul_ps(ret.val[3].0, active_mask, material_color.val[3].0, bounced.val[3].0);

                    // ret.val[0].0 = _mm512_mask_mov_ps(ret.val[0].0, active_mask, _mm512_abs_ps(n.val[0].0));
                    // ret.val[1].0 = _mm512_mask_mov_ps(ret.val[1].0, active_mask, _mm512_abs_ps(n.val[1].0));
                    // ret.val[2].0 = _mm512_mask_mov_ps(ret.val[2].0, active_mask, _mm512_abs_ps(n.val[2].0));
                    // ret.val[3].0 = _mm512_mask_mov_ps(ret.val[3].0, active_mask, _mm512_set1_ps(1.0));
                }
            }

            /*
            if material.emission >= 1. {
                // Do not bounce if fully emissive
                if bounces > 0 {
                    return material.color.mul_scalar(material.emission);
                } else {
                    // To avoid color clipping, don't multiply direct rays by more than 1
                    return material.color;
                }
            } else {
                let mut n = closest_inter.n;
                if n.dot(ray_dir) > 0. {
                    // If normal is facing the wrong direction, flip it
                    n = n.mul_scalar(-1.);
                }

                let point_on_sphere =
                    closest_inter.p.add(random_unit_vector(&mut self.scalar_prng));
                let point_on_sphere_shifted = point_on_sphere.add(n);
                let lambertian_ray_dir =
                    point_on_sphere_shifted.sub(closest_inter.p).normalize();
                // lambertian_ray_dir now should have a lambertian distribution

                return material.color.mul_elementwise(self.zaytrace_ray(
                    maybe_closest_inter_geo_id,
                    closest_inter.p,
                    lambertian_ray_dir,
                    bounces + 1,
                ));
            }
            */

            return ret;
        }
    }
}
