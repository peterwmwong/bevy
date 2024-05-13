#![feature(iter_array_chunks)]
//! Meshlet asset export (experimental).

// Note: This example showcases the exporting a meshlet asset.

use bevy::{
    asset::{io::file::FileAssetReader, saver::AssetSaver},
    pbr::experimental::meshlet::{MeshletMesh, MeshletMeshSaverLoad},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        render_asset::RenderAssetUsages,
    },
    tasks::block_on,
    utils::{hashbrown::HashMap, CowArc},
};
use std::{fs::File, io::BufReader, process::ExitCode};

fn main() -> ExitCode {
    let mut m: Mesh;
    {
        println!(">>> Loading Mesh...");
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
        m = Mesh::new(
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
        println!(">>> Done!");
    }

    let meshlet_mesh;
    {
        println!(">>> Generating Meshlets...");
        meshlet_mesh = MeshletMesh::from_mesh(&m).unwrap();
        println!(">>> Done!");
    };

    {
        println!(">>> Writing...");
        let mut writer = block_on(async_fs::File::create("/tmp/bevy_meshlet.meshlets")).unwrap();
        pub struct FakeSavedAsset<'a, A: Asset> {
            _value: &'a A,
            _labeled_assets: &'a HashMap<CowArc<'static, str>, ()>,
        }
        let fsa = FakeSavedAsset {
            _value: &meshlet_mesh,
            _labeled_assets: &HashMap::new(),
        };
        #[allow(unsafe_code)]
        block_on(MeshletMeshSaverLoad.save(&mut writer, unsafe { core::mem::transmute(fsa) }, &()))
            .unwrap();
        println!(">>> Done!");
    }

    ExitCode::SUCCESS
}
