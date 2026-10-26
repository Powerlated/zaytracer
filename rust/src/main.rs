#![feature(portable_simd)]

use std::sync::Arc;

use std::borrow::Cow;
use ::raytracer::{WINDOW_HEIGHT, WINDOW_WIDTH};
use wgpu::Extent3d;
use wgpu::TextureFormat::Bgra8UnormSrgb;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowButtons, WindowId};

use crate::raytracer::scalar::{MAX_ACCUMULATE_STEPS, RaytracerScalar};
use crate::raytracer::simd::RaytracerSimd;

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
    render_pipeline: wgpu::RenderPipeline,

    texture_data: Box<[u8]>,
    texture: wgpu::Texture,
    texture_bind_group: wgpu::BindGroup,

    raytracer: raytracer::simd::RaytracerSimd,

    raytrace_start_time: Option<std::time::Instant>,
    raytrace_finish_time: Option<std::time::Instant>,
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

        // **Load shaders from disk**
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture.wgsl"))),
        });

        let surface = instance.create_surface(arc_window.clone()).unwrap();

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
            present_mode: wgpu::PresentMode::AutoNoVsync,
        });

        // *Texture setup for drawing raytraced image to screen*
        let texture = device.create_texture(&wgpu::wgt::TextureDescriptor { 
            label: None, 
            size: wgpu::Extent3d {
                width: WINDOW_WIDTH as u32,
                height: WINDOW_HEIGHT as u32,
                depth_or_array_layers: 1
            }, 
            mip_level_count: 1,
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            // Specifies what texture formats can be used to create TextureViews for this texture,
            view_formats: &[]
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: None,
            });

        let texture_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    }
                ],
                label: None,
            }
        );

        // Must add bind group layouts to pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&texture_bind_group_layout)],
            immediate_size: 0
        });

        // And then pipeline layout is used to create the render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render texture from ray tracer"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())]
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        WgpuState {
            instance,
            arc_window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,

            texture_data: vec![0u8; WINDOW_WIDTH * 4 * WINDOW_HEIGHT].into_boxed_slice(),
            texture,
            texture_bind_group,

            raytracer: RaytracerSimd::from_scalar(RaytracerScalar::new()),

            raytrace_start_time: None,
            raytrace_finish_time: None,
        }
    }

    fn render(&mut self) {
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
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.draw(0..6, 0..1);

        // Call draw commands here

        // End the render pass construction. 
        drop(render_pass);

        // Cast self.texture_data into properly typed 2D array
        let u32_slice: &mut [u32] = bytemuck::cast_slice_mut(&mut self.texture_data);
        let flat_array_ref: &mut [u32; WINDOW_WIDTH * WINDOW_HEIGHT] = u32_slice.try_into().unwrap();
        let buffer: &mut [[u32; WINDOW_WIDTH]; WINDOW_HEIGHT] = bytemuck::cast_mut(flat_array_ref);

        if self.raytrace_start_time.is_none() {
            self.raytrace_start_time = Some(std::time::Instant::now());
        }

        // Do the raytrace, one ray per pixel
        // https://docs.amd.com/r/en-US/57368-uProf-user-guide/11.1.5.1.-Group2
        let accumulated_samples = self.raytracer.raytrace_step(buffer);

        if accumulated_samples == MAX_ACCUMULATE_STEPS && self.raytrace_finish_time.is_none() {
            self.raytrace_finish_time = Some(std::time::Instant::now());
        }

        let suffix;
        if let Some(raytrace_finish_time) = self.raytrace_finish_time
         && let Some(raytrace_start_time) = self.raytrace_start_time {
            // If raytracing is done
            suffix = format!("{:.3} s", (raytrace_finish_time - raytrace_start_time).as_secs_f64());
        } else if let Some(raytrace_start_time) = self.raytrace_start_time {
            // If raytracing is ongoing
            suffix = format!("{} s", raytrace_start_time.elapsed().as_secs());
        } else {
            // If raytracing has not started yet
            suffix = "".to_string();
        }

        self.arc_window.set_title(format!("Brian Jia - Raytracer [{}/{}] {}", accumulated_samples, MAX_ACCUMULATE_STEPS, suffix).as_str());

        self.queue.write_texture(
            self.texture.as_image_copy(),
            &self.texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEXTURE_FORMAT.theoretical_memory_footprint(Extent3d { width: WINDOW_WIDTH as u32, height: 1, depth_or_array_layers: 1 }) as u32),
                rows_per_image: Some(WINDOW_HEIGHT as u32),
            },
            self.texture.size(),
        );
    
        // Submit the encoded render pass to the GPU for rendering
        self.queue.submit([encoder.finish()]);

        self.arc_window.pre_present_notify();
        self.queue.present(surface_texture);
    }
}

#[derive(Default)]
struct App {
    wgpu_state: Option<WgpuState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let arc_window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());

        arc_window.set_resizable(false);
        arc_window.set_enabled_buttons(arc_window.enabled_buttons().difference(WindowButtons::MAXIMIZE));
        let hidpi = arc_window.scale_factor() >= 1.5;

        let l_size = if hidpi {
            winit::dpi::PhysicalSize::new((WINDOW_WIDTH * 2) as u32, (WINDOW_HEIGHT * 2) as u32)
        } else {
            winit::dpi::PhysicalSize::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32)
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
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                state.render();
                // Emits a new redraw requested event.
                state.arc_window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    
    // Move this thread onto the highest-numbered logical core
    let core_ids = core_affinity::get_core_ids().unwrap();
    core_affinity::set_for_current(*core_ids.last().unwrap());
    
    // Set this thread to high priority
    thread_priority::set_winapi_thread_priority(
        thread_priority::thread_native_id(), 
        thread_priority::WinAPIThreadPriority::Highest
    ).unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}