#![feature(portable_simd)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(pointer_is_aligned_to)]

mod raytracer_display;

use std::ops::DerefMut;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex};

use std::borrow::Cow;
use std::thread::JoinHandle;
use std::{array, thread};
use std::time::Instant;
use rkyv::vec;
use wgpu::Extent3d;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{KeyCode, KeyLocation, PhysicalKey};
use winit::window::{Window, WindowButtons, WindowId};

use crate::AppMode::Raytrace;
use crate::raytracer::math::Bgraf32;
use crate::raytracer::rng::SFC32;
use crate::raytracer::{RaytracerBuf, new_raytracer_buf};
use crate::raytracer::NUM_WORKERS_SCALAR;
use crate::raytracer::{FORCE_HIDPI, HEIGHT, MAX_ACCUMULATE_STEPS, WIDTH, scalar::*};
use crate::raytracer::simd::*;
use crate::raytracer_display::RaytracerDisplay;
use ::raytracer::GIT_HASH_AND_DIRTY_STR;

#[cfg(not(feature = "simd"))]
const IS_SIMD: bool = false;
#[cfg(feature = "simd")]
const IS_SIMD: bool = true;

mod raytracer;

// **examples this is based off**
// https://github.com/gfx-rs/wgpu/blob/trunk/examples/standalone/02_hello_window/src/main.rs#L3
// https://github.com/gfx-rs/wgpu/blob/trunk/examples/features/src/hello_triangle/mod.rs
struct WgpuState {
    instance: wgpu::Instance,
    arc_window: Arc<winit::window::Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,

    arc_raytracer_scalar: Arc<raytracer::scalar::RaytracerScalar>,
    accumulator_buf: Box<RaytracerBuf>,
    arc_num_samples_accumulated: [Arc<AtomicU32>; NUM_WORKERS_SCALAR],
    arc_lanes_active_pct: Arc<AtomicU32>,
    arc_threads_should_exit: Arc<AtomicBool>,

    thread_handles: Vec<JoinHandle<()>>,

    raytrace_start_time: Option<std::time::Instant>,
    raytrace_finish_time: Option<std::time::Instant>,
    
    raytracer_display: RaytracerDisplay,
}

pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

impl WgpuState {
    async fn init_wgpu(display: OwnedDisplayHandle, arc_window: Arc<Window>) -> WgpuState {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(display)
        ));
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        // This creates the surface that represents the window
        let surface = instance.create_surface(arc_window.clone()).unwrap();

        // *Get preferred swapchain format*
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        // Get preferred render format 
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];
        // let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let inner_size = arc_window.inner_size();
        println!("Creating surface with dimensions {}x{}", inner_size.width, inner_size.height);
        
        let size = arc_window.inner_size();
        // Configure surface for the first time. WebGPU always draws to a texture,
        // just a special one for the main screen
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            // Request compatibility with the sRGB-format texture that will be created later
            view_formats: vec![surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: inner_size.width,
            height: inner_size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        });

        let mut accumulator_buf = new_raytracer_buf();
        let arc_raytracer_scalar = Arc::new(RaytracerScalar::new());
        let arc_num_samples_accumulated = array::from_fn(|_| Arc::new(AtomicU32::new(0)));
        let arc_lanes_active_pct = Arc::new(AtomicU32::new(0));
        let arc_threads_should_exit = Arc::new(AtomicBool::new(false));

        // **Start ray tracer threads**
        let mut thread_handles = Vec::new();
        if IS_SIMD {
            // TODO: get rid of unsafe. the unsafe is to share this buf among threads with no performance penalty
            let accumulator_buf_ptr = accumulator_buf.as_mut_ptr();
            let acc_slice: &mut [Bgraf32] = unsafe {
                std::slice::from_raw_parts_mut(accumulator_buf_ptr, accumulator_buf.len())
            };

            let arc_lanes_active_pct = arc_lanes_active_pct.clone();
            let arc_num_samples_accumulated = arc_num_samples_accumulated[0].clone();
            let arc_threads_should_exit = arc_threads_should_exit.clone();
            
            thread_handles.push(thread::spawn(move || {
                let mut raytracer = RaytracerSimd::<14>::from_scalar(RaytracerScalar::new());
                for i in 0..MAX_ACCUMULATE_STEPS {
                    raytracer.raytrace_step(acc_slice);
                    let lanes_active_pct = raytracer.lanes_active * 100 / raytracer.lanes_computed;
                    arc_lanes_active_pct.store(lanes_active_pct as u32, Relaxed);
                    arc_num_samples_accumulated.store(i+1, Relaxed);

                    if arc_threads_should_exit.load(Relaxed) {
                        return;
                    }
                }
            }));
        } else {
            for i in 0..NUM_WORKERS_SCALAR {
                assert!(HEIGHT % NUM_WORKERS_SCALAR == 0);
                let rows_per_worker = HEIGHT / NUM_WORKERS_SCALAR;
                let pixels_per_worker = rows_per_worker * WIDTH;
                
                let buf_i = i*pixels_per_worker..(i+1)*pixels_per_worker;
                // TODO: get rid of unsafe. the unsafe is to share this buf among threads with no performance penalty
                let accumulator_buf = accumulator_buf[buf_i].as_mut_ptr();
                let acc_slice: &mut [Bgraf32] = unsafe {
                    std::slice::from_raw_parts_mut(accumulator_buf, pixels_per_worker)
                };

                let raytracer = arc_raytracer_scalar.clone();
                let arc_num_samples_accumulated = arc_num_samples_accumulated[i].clone();
                let arc_threads_should_exit = arc_threads_should_exit.clone();

                thread_handles.push(thread::spawn(move || {
                    let mut prng = SFC32::<u32>::seed(i as u32);
                    let rows = i*rows_per_worker..(i+1)*rows_per_worker;
                    
                    for i in 0..MAX_ACCUMULATE_STEPS {
                        raytracer.raytrace_step(rows.clone(), acc_slice, &mut prng);
                        arc_num_samples_accumulated.store(i+1, Relaxed);

                        if arc_threads_should_exit.load(Relaxed) {
                            return;
                        }
                    }
                }));
            }
        }

        let raytracer_display = RaytracerDisplay::new(&device, swapchain_format);

        WgpuState {
            instance,
            arc_window,
            device,
            queue,
            size,
            surface,
            surface_format,


            accumulator_buf,
            arc_raytracer_scalar,
            arc_num_samples_accumulated,
            arc_lanes_active_pct,
            arc_threads_should_exit,

            thread_handles,
            
            raytrace_start_time: None,
            raytrace_finish_time: None,

            raytracer_display
        }
    }

    fn do_raytrace(&mut self) {
        if self.raytrace_start_time.is_none() {
            self.raytrace_start_time = Some(std::time::Instant::now());
        }

        let mut accumulated_samples = self.arc_num_samples_accumulated[0].load(Relaxed);
        if !IS_SIMD {
            // for non-SIMD, the accumulated samples number displayed is from the thread with the fewest samples outputted so far
            accumulated_samples = self.arc_num_samples_accumulated[0].load(Relaxed);
            for i in 1..NUM_WORKERS_SCALAR {
                let val = self.arc_num_samples_accumulated[i].load(Relaxed);
                if val < accumulated_samples {
                    accumulated_samples = val;
                }
            }
        }

        // frontend converts [Bgraf32] buffer into [u8] buffer
        for (i, bgraf32) in self.accumulator_buf.iter().enumerate() {
            let converted = bgraf32.div_scalar(accumulated_samples as f32).to_bgra8888();
            self.raytracer_display.texture_data[i*4+0] = converted[0];  
            self.raytracer_display.texture_data[i*4+1] = converted[1];  
            self.raytracer_display.texture_data[i*4+2] = converted[2];  
            self.raytracer_display.texture_data[i*4+3] = converted[3];  
        }

        if accumulated_samples == MAX_ACCUMULATE_STEPS && self.raytrace_finish_time.is_none() {
            self.raytrace_finish_time = Some(std::time::Instant::now());
        }

        let suffix;
        if let Some(raytrace_finish_time) = self.raytrace_finish_time
         && let Some(raytrace_start_time) = self.raytrace_start_time {
            // If raytracing is done
            suffix = format!(" {:.3} s", (raytrace_finish_time - raytrace_start_time).as_secs_f64());
        } else if let Some(raytrace_start_time) = self.raytrace_start_time {
            // If raytracing is ongoing
            suffix = format!(" {} s", raytrace_start_time.elapsed().as_secs());
        } else {
            // If raytracing has not started yet
            suffix = "".to_string();
        }

        self.arc_window.set_title(format!("Brian Jia - Raytracer{} [{}/{}]{}{}{}", 
            if cfg!(feature = "simd") { " SIMD" } else { " Scalar" },
            accumulated_samples,
            MAX_ACCUMULATE_STEPS,
            suffix,
            format!(" {}", GIT_HASH_AND_DIRTY_STR).as_str(),
            self.statusline().as_str(),
        ).as_str());


        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            other => { unimplemented!() }
        };

        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                // Without add_srgb_suffix() the image we will be working with
                // might not be "gamma correct".
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        // Render a solid black screen
        let mut encoder = self.device.create_command_encoder(&Default::default());
        // Create the render pass object that will clear the screen.
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                }
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.raytracer_display.render_pipeline);
        render_pass.set_bind_group(0, &self.raytracer_display.texture_bind_group, &[]);
        render_pass.draw(0..6, 0..1);

        // Call draw commands here

        // End the render pass construction. 
        drop(render_pass);

        self.queue.write_texture(
            self.raytracer_display.texture.as_image_copy(),
            &*self.raytracer_display.texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEXTURE_FORMAT.theoretical_memory_footprint(Extent3d { width: WIDTH as u32, height: 1, depth_or_array_layers: 1 }) as u32),
                rows_per_image: Some(HEIGHT as u32),
            },
            self.raytracer_display.texture.size(),
        );
    
        // Submit the encoded render pass to the GPU for rendering
        self.queue.submit([encoder.finish()]);

        self.arc_window.pre_present_notify();
        self.queue.present(surface_texture);
    }

    fn do_cool_viz(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            other => { unimplemented!() }
        };

        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                // Without add_srgb_suffix() the image we will be working with
                // might not be "gamma correct".
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        // Render a solid black screen
        let mut encoder = self.device.create_command_encoder(&Default::default());
        // Create the render pass object that will clear the screen.
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                }
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(render_pass);

        self.queue.submit([encoder.finish()]);
        self.arc_window.pre_present_notify();
        self.queue.present(surface_texture);
    }

    fn statusline(&self) -> String {
        if IS_SIMD {
            return format!(" {}% lane util, 1 thread", self.arc_lanes_active_pct.load(Relaxed));
        } else {
            return format!(" {} threads", NUM_WORKERS_SCALAR);
        }
    }
}

enum AppMode {
    Raytrace,
    CoolViz,
}

struct App {
    wgpu_state: Option<WgpuState>,
    app_mode: AppMode,
}

impl App {
    fn new() -> Self {
        Self {
            wgpu_state: None,
            app_mode: AppMode::Raytrace,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let arc_window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());

        arc_window.set_resizable(false);
        arc_window.set_enabled_buttons(arc_window.enabled_buttons().difference(WindowButtons::MAXIMIZE));
        let hidpi = arc_window.scale_factor() >= 1.5 || FORCE_HIDPI;

        let l_size = if hidpi {
            winit::dpi::PhysicalSize::new((WIDTH * 2) as u32, (HEIGHT * 2) as u32)
        } else {
            winit::dpi::PhysicalSize::new(WIDTH as u32, HEIGHT as u32)
        };
        arc_window.set_max_inner_size(Some(l_size));
        arc_window.set_min_inner_size(Some(l_size));
        let _ = arc_window.request_inner_size(l_size);

        self.wgpu_state = Some(pollster::block_on(WgpuState::init_wgpu(
            event_loop.owned_display_handle(), 
            arc_window
        )));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.wgpu_state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; waiting for threads to stop");

                // Tell threads to stop
                self.wgpu_state.as_mut().unwrap().arc_threads_should_exit.store(true, Relaxed);

                // Wait for threads to stop
                for handle in self.wgpu_state.as_mut().unwrap().thread_handles.drain(..) {
                    handle.join().unwrap();
                }

                println!("Thread stopped, exiting");

                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                match self.app_mode {
                    AppMode::Raytrace => {
                        state.do_raytrace();
                    },
                    AppMode::CoolViz => {
                        state.do_cool_viz();
                    }
                }

                // Emits a new redraw requested event.
                state.arc_window.request_redraw();
            },
            WindowEvent::KeyboardInput {device_id: _, event, is_synthetic: _} => {
                if !event.repeat && event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            match self.app_mode {
                                AppMode::CoolViz => {
                                    self.app_mode = AppMode::Raytrace;
                                },
                                AppMode::Raytrace => {
                                    self.app_mode = AppMode::CoolViz;
                                }
                            }
                        },
                        _ => {},
                    }
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    
    // TODO for multithreaded non-SIMD
    #[cfg(feature = "simd")]
    {
        // Move this thread onto the highest-numbered logical core
        let core_ids = core_affinity::get_core_ids().unwrap();
        core_affinity::set_for_current(*core_ids.last().unwrap());
        
        // Set this thread to high priority
        #[cfg(windows)]
        thread_priority::set_winapi_thread_priority(
            thread_priority::thread_native_id(), 
            thread_priority::WinAPIThreadPriority::Highest
        ).unwrap();
    }

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}