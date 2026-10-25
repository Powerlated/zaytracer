// In WebGPU:
//     * NDC +x is to the right,    [-1.0, 1.0]
//     * NDC +y is to the top (up), [-1.0, 1.0]
//     * NDC +z is into the screen, [ 0.0, 1.0]

struct Interstage {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> Interstage {
    var positions = array<vec4f, 3>(
        vec4f( 0.0,  0.5,  0.0, 1.0),
        vec4f(-0.5, -0.5,  0.0, 1.0),
        vec4f( 0.5, -0.5,  0.0, 1.0),
    );

    var colors = array<vec4f, 3>(
        vec4f(1, 0, 0, 1.0),
        vec4f(0, 1, 0, 1.0),
        vec4f(0, 0, 1, 1.0),
    );

    var is: Interstage;
    is.position = positions[in_vertex_index];
    is.color = colors[in_vertex_index];

    return is;
}

@fragment
fn fs_main(is: Interstage) -> @location(0) vec4f {
    return is.color;
}