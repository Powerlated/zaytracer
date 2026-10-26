// In WebGPU:
//     * NDC +x is to the right,    [-1.0, 1.0]
//     * NDC +y is to the top (up), [-1.0, 1.0]
//     * NDC +z is into the screen, [ 0.0, 1.0]
// For textures:
//     * +u is to the right    [0.0, 1.0]
//     * +v is to the top (up) [0.0, 1.0]

struct Interstage {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> Interstage {
    let pos = array(
        // 1st triangle
        vec2f(-1.0, -1.0),  // left,  bottom
        vec2f( 1.0, -1.0),  // right, bottom
        vec2f(-1.0,  1.0),  // left,  top
    
        // 2nd triangle
        vec2f(-1.0,  1.0),  // left,  top
        vec2f( 1.0, -1.0),  // right, bottom
        vec2f( 1.0,  1.0),  // right, top
    );

    let uv = array(
        // 1st triangle
        vec2f( 1.0,  1.0),
        vec2f( 0.0,  1.0),
        vec2f( 1.0,  0.0),
    
        // 2nd triangle
        vec2f( 1.0,  0.0),
        vec2f( 0.0,  1.0),
        vec2f( 0.0,  0.0),
    );

    var is: Interstage;
    let xy = pos[vertexIndex];
    is.position = vec4f(xy, 0.0, 1.0);
    is.texcoord = uv[vertexIndex];
    return is;
}

@group(0) @binding(0) var ourTexture: texture_2d<f32>;
@group(0) @binding(1) var ourSampler: sampler;
 
@fragment fn fs(fsInput: Interstage) -> @location(0) vec4f {
  return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}