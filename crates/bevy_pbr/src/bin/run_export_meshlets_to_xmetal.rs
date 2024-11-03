#![feature(iter_array_chunks)]
//! Meshlet asset export (experimental).

// Note: This example showcases the exporting a meshlet asset.

use bevy_asset::io::file::FileAssetReader;
use bevy_pbr::experimental::meshlet::{export_meshlets_to_xmetal, MinMax};
use bevy_render::{
    mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues},
    render_asset::RenderAssetUsages,
};
use std::{fs::File, io::BufReader, process::ExitCode};

fn main() -> ExitCode {
    let mut m: Mesh;
    let position_denorm_scale;
    {
        println!(">>> Loading Mesh...");
        let model_path = FileAssetReader::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/models/bunny.obj"
        ));
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
            VertexAttributeValues::Float32x3({
                let mut positions: Vec<[f32; 3]> =
                    positions.into_iter().array_chunks::<3>().collect();
                let mut min_max = MinMax::new();
                for p in &positions {
                    min_max.update(p);
                }

                let normalizer = min_max.signed_normalizer();
                position_denorm_scale = normalizer.denorm_scale();
                for p in &mut positions {
                    *p = normalizer.denorm_scale_normalized_position(&normalizer.normalize(p));
                }
                positions
            }),
        );
        m.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals.into_iter().array_chunks::<3>().collect()),
        );
        m.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(texcoords.into_iter().array_chunks::<2>().collect()),
        );
        println!(">>> Done!");
    }

    println!(">>> Generating/Writing Meshlets...");
    export_meshlets_to_xmetal(&m, position_denorm_scale).unwrap();
    println!(">>> Done!");

    ExitCode::SUCCESS
}