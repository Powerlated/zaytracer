use std::sync::Arc;

use std::borrow::Cow;
use wgpu::wgc::command::EncodingApi::Wgpu;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowButtons, WindowId};

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
    render_pipeline: wgpu::RenderPipeline
}

pub const WINDOW_WIDTH: usize = 640;
pub const WINDOW_HEIGHT: usize = 480;

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("triangle.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0
        });

        let surface = instance.create_surface(arc_window.clone()).unwrap();

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())]
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let cap = surface.get_capabilities(&adapter);
        let size = arc_window.inner_size();
        let surface_format = cap.formats[0];

        // Configure surface for the first time
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            // Request compatibility with the sRGB-format texture that will be created later
            view_formats: vec![surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: WINDOW_WIDTH as u32,
            height: WINDOW_HEIGHT as u32,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        });

        WgpuState {
            instance,
            arc_window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline
        }
    }

    fn render(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            _ => { panic!(); }
        };

        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                // Without add_srgb_suffix() the image we will be working with
                // might not be "gamma correct".
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        // Render a solid green screen
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
        render_pass.draw(0..3, 0..1);

        // Call draw commands here

        // End the render pass. 
        drop(render_pass);

        // End the render pass
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
        let l_size = LogicalSize::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32);
        arc_window.set_max_inner_size(Some(l_size));
        arc_window.set_min_inner_size(Some(l_size));
        let _ = arc_window.request_inner_size(l_size);

        self.wgpu_state = Some(pollster::block_on(WgpuState::init_wgpu(
            event_loop.owned_display_handle(), 
            arc_window
        )));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
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