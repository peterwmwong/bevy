#![feature(iter_array_chunks)]
//! Meshlet rendering for dense high-poly scenes (experimental).

// Note: This example showcases the meshlet API, but is not the type of scene that would benefit from using meshlets.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    asset::io::file::FileAssetReader,
    pbr::experimental::meshlet::{MaterialMeshletMeshBundle, MeshletMesh, MeshletPlugin},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::{AsBindGroup, ShaderRef},
    },
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::{f32::consts::PI, fs::File, io::BufReader, process::ExitCode};

fn main() -> ExitCode {
    App::new()
        .add_plugins((
            DefaultPlugins,
            MeshletPlugin,
            MaterialPlugin::<DebugPrimitiveIDMaterial>::default(),
            CameraControllerPlugin,
        ))
        .add_systems(Startup, setup)
        .run();

    ExitCode::SUCCESS
}

fn setup(
    mut commands: Commands,
    _asset_server: Res<AssetServer>,
    mut _standard_materials: ResMut<Assets<StandardMaterial>>,
    mut debug_materials: ResMut<Assets<DebugPrimitiveIDMaterial>>,
    mut meshes: ResMut<Assets<MeshletMesh>>,
) {
    // Load model mesh
    let meshlet_mesh: Handle<MeshletMesh> = {
        let model_path = FileAssetReader::new("assets/models/bunny.obj");
        let model_path = model_path.root_path();
        let mut reader = BufReader::new(
            File::open(model_path)
                .expect(&format!("Failed to open model OBJ file: {model_path:?}")),
        );
        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: false,
                ignore_points: true,
                ignore_lines: true,
            },
            move |_| Ok((vec![], ahash::AHashMap::new())),
        )
        .expect(&format!("Failed to read model OBJ file: {model_path:?}"));
        let tobj::Model {
            mesh:
                tobj::Mesh {
                    indices,
                    positions,
                    normals,
                    texcoords,
                    ..
                },
            ..
        } = models.into_iter().next().expect(&format!(
            "Expected atleast one model in OBJ file: {model_path:?}"
        ));
        let mut m = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        m.insert_indices(Indices::U32(indices));
        m.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(positions.into_iter().array_chunks::<3>().collect()),
        );
        m.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals.into_iter().array_chunks::<3>().collect()),
        );
        m.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(texcoords.into_iter().array_chunks::<2>().collect()),
        );
        m.generate_tangents().unwrap();
        let meshlet_mesh = MeshletMesh::from_mesh(&m).unwrap();
        meshes.add(meshlet_mesh)
    };

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0.5, -0.5))
                .looking_at(Vec3::new(0., 0.1, 0.), Vec3::Y),
            ..default()
        },
        CameraController {
            sensitivity: 0.25,
            walk_speed: 0.5,
            key_forward: KeyCode::KeyE,
            key_back: KeyCode::KeyD,
            key_left: KeyCode::KeyS,
            key_right: KeyCode::KeyF,
            key_up: KeyCode::KeyR,
            key_down: KeyCode::KeyW,
            ..CameraController::default()
        },
    ));

    commands.spawn(MaterialMeshletMeshBundle {
        meshlet_mesh,
        material: debug_materials.add(DebugPrimitiveIDMaterial::default()),
        transform: Transform::default()
            .with_scale(Vec3::splat(0.2))
            .with_rotation(Quat::from_rotation_y(PI))
            .with_translation(Vec3::new(0., 0.0, 0.)),
        ..default()
    });
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct DebugPrimitiveIDMaterial {}
impl Material for DebugPrimitiveIDMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/debug_primitive_id.wgsl".into()
    }
}
