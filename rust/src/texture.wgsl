// In WebGPU:
//     * NDC +x is to the right,    [-1.0, 1.0]
//     * NDC +y is to the top (up), [-1.0, 1.0]
//     * NDC +z is into the screen, [ 0.0, 1.0]

struct Interstage {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> Interstage {
    let pos = array(
        // 1st triangle
        vec2f( 0.0,  0.0),  // center
        vec2f( 1.0,  0.0),  // right, center
        vec2f( 0.0,  1.0),  // center, top
    
        // 2nd triangle
        vec2f( 0.0,  1.0),  // center, top
        vec2f( 1.0,  0.0),  // right, center
        vec2f( 1.0,  1.0),  // right, top
    );

    var is: Interstage;
    let xy = pos[vertexIndex];
    is.position = vec4f(xy, 0.0, 1.0);
    is.texcoord = xy;
    return is;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
 
@fragment fn fs(fsInput: Interstage) -> @location(0) vec4f {
  return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}