use crate::scalar::geometry::transform_in_place;
use crate::*;

#[derive(Clone)]
pub struct Tri {
    // Vertices. typical notation in math is (A, B, C)
    pub v: [Vec3f32; 3],
    pub shader: TriShader,
    // TODO: derive normal direction from winding order?
    // n: Vec3, // n^\hat is normalized normal vector
}

#[derive(Clone)]
pub enum Geo {
    Tri(Tri),
    // sphere: Sphere,
}

pub struct Scene {
    pub geos: Vec<Geo>
}

impl Scene {
    pub fn cornell_box() -> Self {
        let mut geos = Vec::new();

        let mut light = square(1., TriShader::Material(MaterialShader::new_emissive(1., 1., 1., 7.)));
        transform_in_place(&mut light, Mat33f32::rotation(90., 0., 0.).translate(0., 0.999, 1.));
        geos.extend_from_slice(&light);

        // *CORNELL BOX MENTIONED*

        let mut left_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 0., 0.)));
        transform_in_place(&mut left_wall, Mat33f32::rotation(  0.,  90.,  0.).translate( 1., 0., 1.25));
        geos.extend_from_slice(&left_wall);
        
        let mut right_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(0., 1., 0.)));
        transform_in_place(&mut right_wall, Mat33f32::rotation(  0., -90.,  0.).translate(-1., 0., 1.25));
        geos.extend_from_slice(&right_wall);
        
        let mut roof = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));
        transform_in_place(&mut roof, Mat33f32::rotation(  90.,  0.,  0.).translate(0.,  1., 1.25));
        geos.extend_from_slice(&roof);
        
        let mut floor = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));
        transform_in_place(&mut floor, Mat33f32::rotation( -90.,  0.,  0.).translate(0., -1., 1.25));
        geos.extend_from_slice(&floor);

        // try add_quad(Vec3::from_xyz(0, 0.999, 1), 1, 0, 0, 90, .{.material = MaterialShader::new_emissive(0, 0, 1, 1)}); // light

        let mut rear_wall = square(2., TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.)));    // rear wall
        transform_in_place(&mut rear_wall, Mat33f32::rotation(  0.,  0.,  0.).translate(0.,  0., 2.25));
        geos.extend_from_slice(&rear_wall);

        let mut cube = cube(0.5);
        transform_in_place(&mut cube, Mat33f32::rotation(0., -70., 0.).translate(-0.25, -0.75, 1.25));
        geos.extend_from_slice(&cube);

        Self { geos }
    }    
}


fn trongle() -> Geo {
    Geo::Tri(Tri {
            v: [
                Vec3f32::from_xyz(0.5, -0.5, 1.),
                Vec3f32::from_xyz(-0.5, -0.5, 1.),
                Vec3f32::from_xyz(0., 0.5, 1.)
            ],
            shader: TriShader::PerVertex(
                TriPerVertexShader { colors: [
                    [ 1., 0., 0., 0.5 ],
                    [ 0., 1., 0., 0.5 ],
                    [ 0., 0., 1., 0.5 ]
                ]}
            )
        },
    )
}

/// Square with side length `len`, centered on (0, 0, 0), normal facing (0, 0, 1).
pub fn square(len: f32, shader: TriShader) -> [Geo; 2] {
    let half_len = len;

    let t1 = Tri {
        v: [
            Vec3f32::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3f32::from_xyz(-half_len / 2.,  half_len / 2., 0.),
            Vec3f32::from_xyz( half_len / 2.,  half_len / 2., 0.)
        ],
        shader: shader.clone(),
    };

    let t2 = Tri {
        v: [
            Vec3f32::from_xyz( half_len / 2., -half_len / 2., 0.),
            Vec3f32::from_xyz(-half_len / 2., -half_len / 2., 0.),
            Vec3f32::from_xyz( half_len / 2.,  half_len / 2., 0.),
        ],
        shader: shader.clone(),
    };

    let g1 = Geo::Tri(t1);
    let g2 = Geo::Tri(t2);

    return [g1, g2];
}
    
/// Cube with side length `len`, centered on (0, 0, 0).
pub fn cube(len: f32) -> [Geo; 12] {
    let f0 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  x
    let f1 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  y
    let f2 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); //  z
    let f3 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -x
    let f4 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -y
    let f5 = square(len, TriShader::Material(MaterialShader::new_diffuse(1., 1., 1.))); // -z   

    let mut faces = [
        f0[0].clone(), f0[1].clone(),
        f1[0].clone(), f1[1].clone(),
        f2[0].clone(), f2[1].clone(),
        f3[0].clone(), f3[1].clone(),
        f4[0].clone(), f4[1].clone(),
        f5[0].clone(), f5[1].clone()
    ];

    transform_in_place(&mut faces[0..2  ], Mat33::rotation(  0., -90.,  0.).translate( len/2.,      0.,      0.)); //  x
    transform_in_place(&mut faces[2..4  ], Mat33::rotation(-90.,   0.,  0.).translate(     0.,  len/2.,      0.)); //  y
    transform_in_place(&mut faces[4..6  ], Mat33::rotation(  0.,   0.,  0.).translate(     0.,      0.,  len/2.)); //  z
    transform_in_place(&mut faces[6..8  ], Mat33::rotation(  0.,  90.,  0.).translate(-len/2.,      0.,      0.)); // -x
    transform_in_place(&mut faces[8..10 ], Mat33::rotation( 90.,   0.,  0.).translate(     0., -len/2.,      0.)); // -y
    transform_in_place(&mut faces[10..12], Mat33::rotation(180.,   0.,  0.).translate(     0.,      0., -len/2.)); // -z

    return faces;
}
