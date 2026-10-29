#![allow(incomplete_features)]

pub mod avx512;
pub mod geometry;
pub mod rng_portable_simd;

use std::marker::ConstParamTy;
use std::arch::x86_64::*;
// use std::arch::x86_64::*;
use std::simd::u32x16;

use rand::rngs::Xoshiro128PlusPlus;
use rand::Rng;

use crate::raytracer::math::{
    Bgra, Bgraf32, MulAdd, Vec3, Vec3f32, deg_to_rad,
};
use crate::raytracer::scalar::RaytracerScalar;
use crate::raytracer::scalar::geometry::Geo;
use crate::raytracer::scalar::shader::MaterialShader;
use crate::raytracer::simd::avx512::*;
use crate::raytracer::simd::geometry::TriSimd;
use crate::raytracer::simd::rng_portable_simd::SFC32Simd;
use crate::raytracer::*;

// 2-tuple with each entry being random value in [-0.5, 0.5]
// TODO: I heard that random jitter causes a non-uniform sampling distribution (clumping)
#[inline]
fn random_jitter(prng: &mut SFC32Simd) -> (F32x16, F32x16) {
    /// `int`: lower 23 bits must be I.I.D.
    /// Returns random float in [-0.5, 0.5].
    #[inline]
    fn float_hack_negpoint5_to_point5(int: __m512i) -> __m512 {
        unsafe {
            let mantissa_mask = _mm512_set1_epi32(0b11111111111111111111111);
            let mantissa = _mm512_and_epi32(int, mantissa_mask);
            let float_template = _mm512_set1_epi32(0b0_01111111_00000000000000000000000u32 as i32);

            // f1 in [1.0, 2.0)
            let f1: __m512 = _mm512_castsi512_ps(_mm512_or_epi32(mantissa, float_template));
            // minus 1.5 to get [-0.5, 0.5)
            let f2 =_mm512_sub_ps(f1, _mm512_set1_ps(1.5));
            f2
        }
    }

    let x_jitter_int = prng.next_u32();
    let y_jitter_int = prng.next_u32();

    return (
        float_hack_negpoint5_to_point5(__m512i::from(x_jitter_int)).into(),
        float_hack_negpoint5_to_point5(__m512i::from(y_jitter_int)).into(),
    );
}

struct RandomUnitVectorGenerator {
    prng: SFC32Simd,
    buf: [[f32; 32]; 3],
    buf_pos: usize,
}

impl RandomUnitVectorGenerator {
    fn new() -> Self {
        Self {
            prng: SFC32Simd::seed(u32x16::from(
                std::array::from_fn::<u32, 16, _>(|i| i as u32),
            )),
            buf: [[0.0; 32]; 3],
            buf_pos: 0,
        }
    }

    // 3D unit vector that is evenly distributed on the unit sphere
    #[inline]
    fn next(&mut self) -> Vec3<F32x16> {
        loop {
            // get three random values - 96 bits of randomness per vector
            let xyz_int: [u32x16; 3] = [
                self.prng.next_u32(),
                self.prng.next_u32(),
                self.prng.next_u32(),
            ];

            // evil floating point bit level magic hacking
            //  1 bit  8 bits    (1).23 bits          
            //  sign   exponent  (implicit 1).mantissa
            // * mantissa has implicit leading 1 bit, giving 24 bits of effective precision
            // * value = (-1)^sign * (1.0 + fraction) * 2^(exponent-127)
            
            /// `int`: lower 23 bits must be I.I.D.
            /// Returns random float in [-1, 1].
            #[inline]
            fn float_hack_neg1_to_1(int: __m512i) -> __m512 {
                unsafe {
                    let mantissa_mask = _mm512_set1_epi32(0b11111111111111111111111);
                    let mantissa = _mm512_and_epi32(int, mantissa_mask);
                    let float_template = _mm512_set1_epi32(0b0_10000000_00000000000000000000000u32 as i32);

                    // f1 in [2.0, 4.0)
                    let f1: __m512 = _mm512_castsi512_ps(_mm512_or_epi32(mantissa, float_template));
                    // minus three to get [-1.0, 1.0)
                    let f2 =_mm512_sub_ps(f1, _mm512_set1_ps(3.0));
                    f2
                }
            }

            let xyz_float = Vec3::<F32x16>::from_xyz(
                F32x16::from(float_hack_neg1_to_1(xyz_int[0].into())),
                F32x16::from(float_hack_neg1_to_1(xyz_int[1].into())),
                F32x16::from(float_hack_neg1_to_1(xyz_int[2].into())),
            );

            let inv_xyz_magn = xyz_float.inv_magnitude_rsqrt14();
            // vectors with magnitude less than 1 and greater than 0.001 are valid
            // -> this is algebraically equivalent to xyz_magn.cmp_le(1.0) & xyz_magn.cmp_ge(0.001);
            let valid_mask = inv_xyz_magn.cmp_ge(1.0) & inv_xyz_magn.cmp_le(31.6227766017);

            // after this, all vectors that satisfy the valid condition will be I.I.D. unit vectors
            let xyz_float = xyz_float.mul_scalar(inv_xyz_magn);

            // write the valid unit vectors into the output buffer
            unsafe {
                // compressstore is slow on Zen 4: https://lemire.me/blog/2025/02/14/avx-512-gotcha-avoid-compressing-words-to-memory-with-amd-zen-4-processors/
                xyz_float.val[0].compressstoreu((&mut self.buf[0][self.buf_pos]) as *mut f32, valid_mask);
                xyz_float.val[1].compressstoreu((&mut self.buf[1][self.buf_pos]) as *mut f32, valid_mask);
                xyz_float.val[2].compressstoreu((&mut self.buf[2][self.buf_pos]) as *mut f32, valid_mask);
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

const ACCUMULATOR_BUF_LEN: usize = WIDTH * HEIGHT;
const INTERMEDIATE_BUF_LEN: usize = ACCUMULATOR_BUF_LEN;
const RAY_RESTORE_DATA_LEN: usize = WIDTH * HEIGHT;

fn vecboxslice<const L: usize, T: Clone + std::fmt::Debug>(val: T) -> Box<[T; L]> {
    return vec![val; L].into_boxed_slice().try_into().unwrap();
}

#[repr(align(64))]
struct RayRestoreData {
    // ray_color: Bgra<Box<[f32; RAY_RESTORE_DATA_LEN]>>,
    origin_id: Box<[u32; RAY_RESTORE_DATA_LEN]>,
    ray_pos: Vec3<Box<[f32; RAY_RESTORE_DATA_LEN]>>, 
    ray_dir: Vec3<Box<[f32; RAY_RESTORE_DATA_LEN]>>, 
    bounces: Box<[u32; RAY_RESTORE_DATA_LEN]>,
    row: Box<[u32; RAY_RESTORE_DATA_LEN]>,
    col: Box<[u32; RAY_RESTORE_DATA_LEN]>,
    entries: u32
}

impl RayRestoreData {
    fn zero() -> Self {
        Self {
            origin_id: vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0),
            ray_pos: Vec3::<Box::<[f32; RAY_RESTORE_DATA_LEN]>> {
                val: [
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                ]
            },
            ray_dir: Vec3::<Box::<[f32; RAY_RESTORE_DATA_LEN]>> {
                val: [
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                    vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0.0),
                ]
            },
            bounces: vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0),
            row: vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0),
            col: vecboxslice::<RAY_RESTORE_DATA_LEN, _>(0),

            entries: 0
        }
    }
}

struct SecondaryBuffer {
    bufs: [RayRestoreData; 2],
    buf_i: usize
}

impl SecondaryBuffer {
    fn new() -> Self {
        Self { 
            bufs: [ RayRestoreData::zero(), RayRestoreData::zero() ],
            buf_i: 0,
        }
    }

    fn get_read(&self) -> &RayRestoreData {
        return &self.bufs[self.buf_i & 1];
    }

    fn get_read_mut(&mut self) -> &mut RayRestoreData {
        return &mut self.bufs[self.buf_i & 1];
    }

    fn get_write(&mut self) -> &mut RayRestoreData {
        return &mut self.bufs[(self.buf_i & 1) ^ 1];
    }

    fn swap_buffers(&mut self) {
        self.buf_i ^= 1;
    }
}

pub struct RaytracerSimd<const REBUNDLE_BELOW: u32> {
    main_prng: SFC32Simd,
    random_unit_vector_gen: RandomUnitVectorGenerator,
    geos: Vec<TriSimd>,
    intermediate_buf: Bgra<Box<[f32; INTERMEDIATE_BUF_LEN]>>,
    accumulator_buf: Bgra<Box<[f32; INTERMEDIATE_BUF_LEN]>>,
    accumulate_steps: u32,

    secondary_buf: SecondaryBuffer,

    pub lanes_active: u64,
    pub lanes_computed: u64,
}

impl<const REBUNDLE_BELOW: u32> RaytracerSimd<REBUNDLE_BELOW> {
    pub fn from_scalar(raytracer_scalar: RaytracerScalar) -> Self {
        let mut geos = Vec::new();
        let mut shaders = Vec::new();

        // geos[0] is placeholder, with its shader values being used for background color
        geos.push(TriSimd {
            v: [Vec3f32::zero(); 3],
            v2_sub_v0: Vec3f32::zero(),
            v1_sub_v0: Vec3f32::zero(),
            n: Vec3f32::zero(),
            shader: MaterialShader {
                color: Bgraf32::zero(),
                emission: 0.0
            }
        });
        // shaders[0] is backgrond color
        shaders.push(MaterialShader { color: Bgraf32::from_rgb_f32(0., 0., 0.), emission: 0.0 });

        for geo in raytracer_scalar.geos {
            match geo {
                Geo::Tri(tri) => {
                    geos.push(TriSimd::from_scalar_tri(tri));
                }
                _ => unimplemented!(),
            }
        }

        Self {
            // start the PRNGs from consistent seed always
            main_prng: SFC32Simd::seed(u32x16::from(
                std::array::from_fn::<u32, 16, _>(|i| i as u32),
            )),
            random_unit_vector_gen: RandomUnitVectorGenerator::new(),
            geos,
            accumulator_buf: Bgra::<Box::<[f32; INTERMEDIATE_BUF_LEN]>> { val: [
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
            ]},
            accumulate_steps: 0,

            lanes_active: 0,
            lanes_computed: 0,

            secondary_buf: SecondaryBuffer::new(),

            intermediate_buf: Bgra::<Box::<[f32; INTERMEDIATE_BUF_LEN]>> { val: [
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
                vecboxslice::<INTERMEDIATE_BUF_LEN, _>(0.0),
            ]}
        }
    }

    #[inline]
    fn update_and_dither_output_buffer(&mut self, buf: &mut RaytracerBuf) {
        let inv_accumulate_steps = 1. / self.accumulate_steps as f32;

        // TODO: dithering
        // step by 16 pixels at a time
        for i in (0..ACCUMULATOR_BUF_LEN).step_by(16) {
            // TODO: Vectorize this
            for j in 0..16 {
                // 4 bytes per pixel in output buffer
                let offs = i+j;
                let offs_out = offs*4;
                let out_pixel = &mut buf[offs_out..offs_out+4];
                let in_b = self.accumulator_buf.val[0][offs] * inv_accumulate_steps; 
                let in_g = self.accumulator_buf.val[1][offs] * inv_accumulate_steps; 
                let in_r = self.accumulator_buf.val[2][offs] * inv_accumulate_steps; 
                out_pixel[0] = in_b.clamp(0.0, 1.0).mul_add(255.0, 0.5) as u8;
                out_pixel[1] = in_g.clamp(0.0, 1.0).mul_add(255.0, 0.5) as u8;
                out_pixel[2] = in_r.clamp(0.0, 1.0).mul_add(255.0, 0.5) as u8;
                out_pixel[3] = 0xFF;
            }
        }

        // let mut dither_prng =
            // Xoshiro128PlusPlusSimd::<16>::seed_from_u64(u64x16::from(std::array::from_fn::<u64, 16, _,>(|i| i as u64)));

        // 16 pixels
        // let out_buf: &mut [[u8; 16 * 4]] = bytemuck::cast_slice_mut(buf);
        // // An Rgbf32::<F32x16> is 16 pixels

        // // 16 pixels per iteration
        // for (acc, out) in self.accumulator_buf.iter().zip(out_buf.iter_mut()) {
        //     let int = dither_prng.next_u32();

        //     let uniform_rect_dist =
        //         (F32x16::from(int) / (0xFFFFFFFFu32 as f32) - 0.5) * (1. / (DITHER_DIVISOR as f32));

        //     let inv_accumulate_steps: F32x16 = (1.0 / self.accumulate_steps as f32).into();
        //     let dithered_f32_b = acc.val[0] * inv_accumulate_steps + uniform_rect_dist;
        //     let dithered_f32_g = acc.val[1] * inv_accumulate_steps + uniform_rect_dist;
        //     let dithered_f32_r = acc.val[2] * inv_accumulate_steps + uniform_rect_dist;
        //     let scaled_f32_a = acc.val[3] * inv_accumulate_steps;

        //     // // Construct the first 4 pixels in f32
        //     // unsafe {
        //     //     // Construct BG
        //     //     // let permute = _mm512_permutex2var_ps;
        //     //     // idx[i+3:i] is the index into the source vectors
        //     //     // idx[i+4] should be 1 to select b
        //     //     let indices: [u8; 16] = [
        //     //         0,
        //     //         0,
        //     //         0b00000,
        //     //         0b10000,
        //     //         0,
        //     //         0,
        //     //         0b00001,
        //     //         0b10001,
        //     //         0,
        //     //         0,
        //     //         0b00010,
        //     //         0b10010,
        //     //         0,
        //     //         0,
        //     //         0b00011,
        //     //         0b10011,
        //     //     ];
        //     //     let indices = _mm_loadu_epi8(indices.as_ptr() as *const i8);
        //     //     let indices = _mm512_cvtepi8_epi32(indices);
        //     //     let mask: u16 = 0b1100110011001100;
        //     //     _mm512_mask2_permutex2var_ps(dithered_f32_b.0, indices, mask, dithered_f32_g.0);
        //     // }

        //     let dithered_f32_b_arr: [f32; 16] = dithered_f32_b.into();
        //     let dithered_f32_g_arr: [f32; 16] = dithered_f32_g.into();
        //     let dithered_f32_r_arr: [f32; 16] = dithered_f32_r.into();

        //     // TODO: Vectorize this
        //     for i in 0..16 {
        //         let out_pixel = &mut out[i * 4..i * 4 + 4];
        //         out_pixel[0] = ((dithered_f32_b_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
        //         out_pixel[1] = ((dithered_f32_g_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
        //         out_pixel[2] = ((dithered_f32_r_arr[i].clamp(0.0, 1.0) * 255.0) + 0.5) as u8;
        //         out_pixel[3] = 0xFF;
        //     }

        //     // let color_255: U8x16 = ((dithered_f32.clamp(0.0, 1.0) * 255.0) + 0.5).into();
        // }
    }

    pub fn raytrace_to_completion(&mut self, buf: &mut RaytracerBuf) {
        while self.accumulate_steps < MAX_ACCUMULATE_STEPS {
            self.raytrace_step(buf);
        }
    }

    #[inline(never)]
    pub fn raytrace_step(&mut self, buf: &mut RaytracerBuf) -> u32 {
        if self.accumulate_steps < MAX_ACCUMULATE_STEPS {
            // **Send out primary rays**
            for row in 0..HEIGHT {
                // map row to normalized y coordinates
                //     (Vulkan style, [-1, -1] xy is the top left corner of the screen,
                //      and z points into the screen, away from the viewer.)

                for col_stepped in (0..WIDTH).step_by(16) {
                    // **First, trace one ray per pixel, and then update the accumulator
                    // buffer with the cumulative average.**

                    let col_offs = U32x16::from(&std::array::from_fn::<_, 16, _>(|i| i as u32));
                    let col = U32x16::from(col_stepped as u32) + col_offs;

                    let mut row_jittered = F32x16::from(row as f32);
                    let mut col_jittered = F32x16::from(col);
                    if ENABLE_JITTER {
                        let jitter = random_jitter(&mut self.main_prng);
                        row_jittered = row_jittered + jitter.0;
                        col_jittered = col_jittered + jitter.1;
                    }

                    pub fn map_range(s: F32x16, a1: f32, a2: f32, b1: f32, b2: f32) -> F32x16 {
                        // original: b1 + (s - a1) * (b2 - b1) / (a2 - a1)
                        //                           <-------const------->
                        return F32x16::mul_add(s - F32x16::splat(a1), 
                                               F32x16::splat((b2 - b1) / (a2 - a1)), 
                                               F32x16::splat(b1));
                    }

                    let norm_y = map_range(row_jittered, 0.0, HEIGHT as f32 - 1.0, -1.0, 1.0);
                    let norm_x = map_range(col_jittered, 0.0, WIDTH as f32 - 1.0, -1.0, 1.0);

                    self.zaytrace_primary_rays(norm_x, norm_y, row as u32, col, col_stepped as u32);
                }
            }

            self.zaytrace_secondary_rays();

            unsafe {
                // **Accumulate intermediate buffer into accumulator buffer**
                for i in (0..INTERMEDIATE_BUF_LEN).step_by(16) {
                    let b_inter_ptr = self.intermediate_buf.val[0].as_mut_ptr().add(i);
                    let g_inter_ptr = self.intermediate_buf.val[1].as_mut_ptr().add(i);
                    let r_inter_ptr = self.intermediate_buf.val[2].as_mut_ptr().add(i);
                    let b_acc_ptr = self.accumulator_buf.val[0].as_mut_ptr().add(i);
                    let g_acc_ptr = self.accumulator_buf.val[1].as_mut_ptr().add(i);
                    let r_acc_ptr = self.accumulator_buf.val[2].as_mut_ptr().add(i);

                    F32x16::store(b_acc_ptr, F32x16::load(b_inter_ptr) + F32x16::load(b_acc_ptr));
                    F32x16::store(g_acc_ptr, F32x16::load(g_inter_ptr) + F32x16::load(g_acc_ptr));
                    F32x16::store(r_acc_ptr, F32x16::load(r_inter_ptr) + F32x16::load(r_acc_ptr));
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
    #[inline]
    fn zaytrace_primary_rays(&mut self, norm_x: F32x16, flipped_norm_y: F32x16, row: u32, col: U32x16, col_stepped: u32) {
        // Flip y to make y up
        let norm_y = flipped_norm_y * -1.;

        // **Perspective Projection**
        const FOV_HALF_HORIZONTAL_RAD: f32 = deg_to_rad(70. / 2.);

        // Given a projection plane 1 unit ahead of the camera,
        // how far away from the center of the plane are the
        // left/right (x_outer_bound) and top/bottom (y_outer_bound) edges?
        let x_outer_bound = f32::tan(FOV_HALF_HORIZONTAL_RAD);
        let y_outer_bound = x_outer_bound * ((HEIGHT as f32) / (WIDTH as f32));

        let x_dir_unnormalized = norm_x * x_outer_bound;
        let y_dir_unnormalized = norm_y * y_outer_bound;

        let ray_pos = Vec3f32::from_xyz(0.0, 0.0, -2.0);
        // TODO: is normalizing necessary?
        let ray_dir = Vec3::<F32x16>::from_xyz(x_dir_unnormalized, y_dir_unnormalized, 1.0.into()).normalize_rsqrt14();

        let active_mask = 0xFFFF;
        let init_ray_color = Bgra::<F32x16>::from_rgb_f32(1., 1., 1.);
        let ray_color = self.zaytrace_bounce(
            init_ray_color,
            U32x16::from(0),
            ray_pos.splat(),
            ray_dir,
            active_mask,
            0,
            U32x16::splat(row),
            col,
            true
        );

        // write primary rays to intermediate buffer
        unsafe {
            // operating on planar format here, 4 bytes per pixel
            let offset_row = row as usize * WIDTH; 
            let offset = offset_row + col_stepped as usize;
            F32x16::store(self.intermediate_buf.val[0].as_mut_ptr().add(offset), ray_color.val[0]);
            F32x16::store(self.intermediate_buf.val[1].as_mut_ptr().add(offset), ray_color.val[1]);
            F32x16::store(self.intermediate_buf.val[2].as_mut_ptr().add(offset), ray_color.val[2]);
        }
    }

    #[inline]
    fn zaytrace_secondary_rays(&mut self) {
        unsafe {
            for i in 1..MAX_BOUNCES {
                self.secondary_buf.swap_buffers();
                // println!("Processing {} secondary rays", self.secondary_buf.get_read().entries);
                while self.secondary_buf.get_read().entries > 0 {
                    let res = self.secondary_buf.get_read_mut();
                    let empty_slots;
                    if res.entries < 16 {
                        empty_slots = 16 - res.entries;
                        res.entries = 0;
                    } else {
                        empty_slots = 0;
                        res.entries -= 16;
                    }
                    let i = res.entries as usize;
                    let origin_id = U32x16::load(&res.origin_id[i] as *const u32);
                    let ray_pos_x = F32x16::load(&res.ray_pos.val[0][i] as *const f32);
                    let ray_pos_y = F32x16::load(&res.ray_pos.val[1][i] as *const f32);
                    let ray_pos_z = F32x16::load(&res.ray_pos.val[2][i] as *const f32);
                    let ray_dir_x = F32x16::load(&res.ray_dir.val[0][i] as *const f32);
                    let ray_dir_y = F32x16::load(&res.ray_dir.val[1][i] as *const f32);
                    let ray_dir_z = F32x16::load(&res.ray_dir.val[2][i] as *const f32);
                    let row = U32x16::load(&res.row[i] as *const u32);
                    let col = U32x16::load(&res.col[i] as *const u32);

                    // obtain secondary ray color from intermediate buffer
                    let offset = row * U32x16::splat(WIDTH as u32) + col;

                    let secondary_color = Bgra::<F32x16> { val: [
                        F32x16::gather::<4>(self.intermediate_buf.val[0].as_ptr(), offset),
                        F32x16::gather::<4>(self.intermediate_buf.val[1].as_ptr(), offset),
                        F32x16::gather::<4>(self.intermediate_buf.val[2].as_ptr(), offset),
                        F32x16::splat(1.0)
                    ]};

                    let active_mask = 0xFFFF >> empty_slots;
                    let secondary = self.zaytrace_bounce(
                        secondary_color,
                        U32x16::from(origin_id),
                        Vec3::<F32x16>::from_xyz(
                            F32x16::from(ray_pos_x),
                            F32x16::from(ray_pos_y),
                            F32x16::from(ray_pos_z),
                        ),
                        Vec3::<F32x16>::from_xyz(
                            F32x16::from(ray_dir_x),
                            F32x16::from(ray_dir_y),
                            F32x16::from(ray_dir_z),
                        ),
                        active_mask,
                        1,
                        U32x16::from(row),
                        U32x16::from(col),
                        true);

                    // write secondary rays to intermediate buffer
                    // operating on planar format here, 4 bytes per pixel
                    let scatter_mask = 0xFFFF >> empty_slots;
                    secondary.val[0].mask(scatter_mask).scatter::<4>(self.intermediate_buf.val[0].as_mut_ptr(), offset);
                    secondary.val[1].mask(scatter_mask).scatter::<4>(self.intermediate_buf.val[1].as_mut_ptr(), offset);
                    secondary.val[2].mask(scatter_mask).scatter::<4>(self.intermediate_buf.val[2].as_mut_ptr(), offset);
                }

                assert_eq!(self.secondary_buf.get_read().entries, 0);
            }
        }
    }

    // **Inner raytracing loop** :D
    // Traces one bounce of 16 rays in parallel
    #[inline(always)]
    fn zaytrace_bounce(
        &mut self,
        mut ray_color: Bgra::<F32x16>,
        mut origin_id: U32x16,
        mut ray_pos: Vec3<F32x16>,
        mut ray_dir: Vec3<F32x16>,
        mut active_mask: u16, // rays that are still 'alive' i.e. still bouncing around
        mut bounces: u8,
        row: U32x16,
        col: U32x16,
        emit_secondary: bool,
    ) -> Bgra::<F32x16> {
        unsafe {
            loop {
                self.lanes_active += active_mask.count_ones() as u64;
                self.lanes_computed += 16;

                let mut hit_mask: u16 = 0; // rays, that in this current bounce, have hit a triangle
                let mut hit_ids = U32x16::splat(0);
                
                let mut hit_t = F32x16::splat(1000.0);
                let mut hit_pos = Vec3::<F32x16>::zero();

                // **Test rays against all triangles**
                assert!(self.geos.len() > 1);
                for (tri_id, tri) in self.geos.iter().enumerate().skip(1) {
                    let hit = tri.intersect(ray_pos, ray_dir);

                    {
                        let mut write_mask = hit.hitmask & hit.t.cmp_lt(hit_t);
                        
                        // Prevent rays from hitting their origin triangle
                        write_mask &= origin_id.cmp_neq(tri_id as u32);

                        // Update the hit_<x> variables for rays we've found a closer intersection for this iteration
                        hit_t = hit_t.mask(write_mask).mov(hit.t);
                        hit_pos.val[0] = hit_pos.val[0].mask(write_mask).mov(hit.p.val[0]); 
                        hit_pos.val[1] = hit_pos.val[1].mask(write_mask).mov(hit.p.val[1]); 
                        hit_pos.val[2] = hit_pos.val[2].mask(write_mask).mov(hit.p.val[2]); 

                        hit_mask |= write_mask;
                        hit_ids = hit_ids.mask(write_mask).splat(tri_id as u32);
                    }

                    // TODO: alpha
                }
                
                // **Look up surface data for rays: Material color, emissivity, and triangle normal vector **
                let material_color: Bgra<F32x16>;
                let material_emission: F32x16;
                let hit_n: Vec3<F32x16>;
                {
                    let tri_size = U32x16::splat(size_of::<TriSimd>() as u32);
                    let tri_offset = hit_ids * tri_size;

                    // Look up color. Blue, green, and then red
                    material_color = Bgra::<F32x16> { val: std::array::from_fn(|i| {
                        F32x16::gather::<1>(&self.geos[0].shader.color.val[i] as *const f32, tri_offset)
                    })};
                    
                    // Look up emission.
                    material_emission = F32x16::gather::<1>(&self.geos[0].shader.emission as *const f32, tri_offset); 

                    // Look up triangle normal
                    hit_n = Vec3::<F32x16> { val: std::array::from_fn(|i| {
                        F32x16::gather::<1>(&self.geos[0].n.val[i] as *const f32, tri_offset)
                    })};
                }

                // TODO: support non-fully emissive
                // handle rays that escaped into the background (!hit_mask) as an emissive hit for material 0
                let emissive_mask = active_mask & (!hit_mask | material_emission.cmp_ge(1.0));
                
                // **Handle emission**
                if emissive_mask != 0 {
                    // Do not bounce if fully emissive
                    active_mask &= !emissive_mask;

                    if bounces > 0 {
                        for i in 0..3 {
                            ray_color.val[i] = ray_color.val[i].mask(emissive_mask).mul(material_color.val[i] * material_emission);
                        }
                    } else {
                        // To avoid color clipping, don't apply gain to direct rays
                        for i in 0..3 {
                            ray_color.val[i] = ray_color.val[i].mask(emissive_mask).mul(material_color.val[i]);
                        }
                    }
                }

                // **Handle diffuse**
                // Bounce if there are still rays active 
                if active_mask != 0 {
                    let mut n = hit_n;

                    // If normal is facing the wrong direction, flip sign
                    {
                        let normal_facing_wrong_direction_mask = n.dot(ray_dir).cmp_gt(0.0);
                        // Evil floating point bit level magic hacking??????
                        let sign_bit = F32x16(std::mem::transmute(_mm512_set1_epi32(0x80000000u32 as i32)));
                        n.val[0] = n.val[0].mask(normal_facing_wrong_direction_mask).xor(sign_bit);
                        n.val[1] = n.val[1].mask(normal_facing_wrong_direction_mask).xor(sign_bit);
                        n.val[2] = n.val[2].mask(normal_facing_wrong_direction_mask).xor(sign_bit);
                    }

                    let point_on_sphere = hit_pos.add(self.random_unit_vector_gen.next());
                    let point_on_sphere_shifted = point_on_sphere.add(n);
                    // TODO: is normalizing necessary?
                    let lambertian_ray_dir = point_on_sphere_shifted.sub(hit_pos).normalize_rsqrt14();

                    ray_color.val[0] = ray_color.val[0].mask(active_mask).mul(material_color.val[0]);
                    ray_color.val[1] = ray_color.val[1].mask(active_mask).mul(material_color.val[1]);
                    ray_color.val[2] = ray_color.val[2].mask(active_mask).mul(material_color.val[2]);

                    // Update rays to new location
                    origin_id = hit_ids;
                    ray_pos = hit_pos;
                    ray_dir = lambertian_ray_dir;

                    bounces += 1;
                    if bounces > MAX_BOUNCES {
                        // If at this point, the ray still has not hit a light (still active), set the ray to black
                        ray_color.val[0] = ray_color.val[0].mask(active_mask).splat(0.0);
                        ray_color.val[1] = ray_color.val[1].mask(active_mask).splat(0.0);
                        ray_color.val[2] = ray_color.val[2].mask(active_mask).splat(0.0);

                        return ray_color;
                    }

                    // TODO
                    if active_mask.count_ones() < REBUNDLE_BELOW && emit_secondary {
                        // If ray vector becomes sparse, write restore data for next pass 
                        let res = self.secondary_buf.get_write();
                        let i = res.entries as usize;

                        origin_id.compressstoreu(&mut res.origin_id[i] as *mut u32, active_mask);
                        ray_pos.val[0].compressstoreu(&mut res.ray_pos.val[0][i] as *mut f32, active_mask);
                        ray_pos.val[1].compressstoreu(&mut res.ray_pos.val[1][i] as *mut f32, active_mask);
                        ray_pos.val[2].compressstoreu(&mut res.ray_pos.val[2][i] as *mut f32, active_mask);
                        ray_dir.val[0].compressstoreu(&mut res.ray_dir.val[0][i] as *mut f32, active_mask);
                        ray_dir.val[1].compressstoreu(&mut res.ray_dir.val[1][i] as *mut f32, active_mask);
                        ray_dir.val[2].compressstoreu(&mut res.ray_dir.val[2][i] as *mut f32, active_mask);
                        row.compressstoreu(&mut res.row[i] as *mut u32, active_mask);
                        col.compressstoreu(&mut res.col[i] as *mut u32, active_mask);

                        res.entries += active_mask.count_ones() as u32;

                        return ray_color;
                    }
                } else {
                    // Vector has no more alive rays
                    return ray_color;
                }
            }
        }
    }
}
