It's a ray tracer that I'm making for fun!

Working on optimizing for AVX512 right now, using Rust portable SIMD!

# Timings

Profiling setup (Windows):
* Modify `AMD_UPROF_PATH` in `build.rs`

Timings are recorded on a Dell Inspiron 16 5645 laptop with an AMD Ryzen 7 8840U CPU, with 40 GB of RAM at 4800 MT/s. 16 GB of 40 GB are in dual-channel mode, the rest are in single-channel mode due to differing RAM stick sizes and speeds (8 GB + 32 GB).

128 rays per pixel are used.

The scene rendered for timings:

![Scene](assets/screenshot-8ab06df.png)

| Commit Hash | Runtime  | Millions of cycles to trace one step | Notes                                                               |
| ----------- | -------- | ------------------------------------ | ------------------------------------------------------------------- |
| 8ab06df     | 32.846 s | 1226                                 | No BVH, no SoA, autovectorization enabled with target-cpu-x86-64-v4 |
| 8ab06df     | N/A      | 1351                                 | Same as above, but on target-cpu-x86-64                             |
| 88b3b85     | N/A      | 474                                  | Initial SIMD attempt                                                |
| 545dded     | 11.413 s | N/A                                  | SIMD with a lot of `fn`s marked `#[inline]`; 41% lane utilization   |