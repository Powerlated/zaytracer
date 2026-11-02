use std::borrow::Cow;

use wgpu::Extent3d;

use crate::TEXTURE_FORMAT;


pub struct RaytracerDisplay {
    pub render_pipeline: wgpu::RenderPipeline,
    pub texture_data: Box<[u8; crate::raytracer::WIDTH*crate::raytracer::HEIGHT*4]>,
    pub texture: wgpu::Texture,
    pub texture_bind_group: wgpu::BindGroup,
}

impl RaytracerDisplay {
    pub fn new(device: &wgpu::Device, render_target_format: wgpu::TextureFormat) -> Self {
        // **Load shaders from disk**
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture.wgsl"))),
        });

        // *Texture setup for drawing raytraced image to screen*
        let texture = device.create_texture(&wgpu::wgt::TextureDescriptor { 
            label: None, 
            size: wgpu::Extent3d {
                width: crate::raytracer::WIDTH as u32,
                height: crate::raytracer::HEIGHT as u32,
                depth_or_array_layers: 1
            }, 
            mip_level_count: 1,
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2,
            format: crate::TEXTURE_FORMAT,
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
                targets: &[Some(render_target_format.into())]
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            render_pipeline,
            texture_data: vec![0u8; crate::raytracer::WIDTH*crate::raytracer::HEIGHT*4].into_boxed_slice().try_into().unwrap(),
            texture,
            texture_bind_group
        }
    }
}