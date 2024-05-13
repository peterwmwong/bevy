#![feature(iter_array_chunks)]
//! Meshlet rendering for dense high-poly scenes (experimental).

// Note: This example showcases the meshlet API, but is not the type of scene that would benefit from using meshlets.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    asset::AssetLoader,
    pbr::experimental::meshlet::{
        MaterialMeshletMeshBundle, MeshletMesh, MeshletMeshSaverLoad, MeshletPlugin,
    },
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderRef},
    tasks::block_on,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::{f32::consts::PI, process::ExitCode};

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
        let mut reader = block_on(async_fs::File::open("/tmp/bevy_meshlet.meshlets")).unwrap();
        let meshlet_mesh = block_on(<MeshletMeshSaverLoad as AssetLoader>::load(
            &MeshletMeshSaverLoad,
            &mut reader,
            &(),
            #[allow(unsafe_code, deref_nullptr)]
            unsafe {
                &mut *core::ptr::null_mut()
            },
        ))
        .unwrap();
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
