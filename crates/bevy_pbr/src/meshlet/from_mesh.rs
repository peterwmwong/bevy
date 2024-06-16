use crate::meshlet::min_max::MinMax;

use super::{
    asset::{Meshlet, MeshletBoundingSphere, MeshletBoundingSpheres, MeshletMesh},
    EncodedVertexPosition, ModelMetadata,
};
use bevy_math::Vec3;
use bevy_render::{
    mesh::{Indices, Mesh},
    render_resource::PrimitiveTopology,
};
use bevy_utils::{HashMap, HashSet};
use itertools::Itertools;
use meshopt::{
    build_meshlets, compute_cluster_bounds, compute_meshlet_bounds,
    ffi::{meshopt_Bounds, meshopt_optimizeMeshlet},
    simplify, simplify_scale, Meshlets, SimplifyOptions, VertexDataAdapter,
};
use metis::Graph;
use std::{borrow::Cow, ops::Range, path::Path, sync::Arc};

impl MeshletMesh {
    /// Process a [`Mesh`] to generate a [`MeshletMesh`].
    ///
    /// This process is very slow, and should be done ahead of time, and not at runtime.
    ///
    /// This function requires the `meshlet_processor` cargo feature.
    ///
    /// The input mesh must:
    /// 1. Use [`PrimitiveTopology::TriangleList`]
    /// 2. Use indices
    /// 3. Have the exact following set of vertex attributes: `{POSITION, NORMAL, UV_0, TANGENT}`
    pub fn from_mesh(
        mesh: &Mesh,
        denorm_scale: [f32; 3],
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        // Validate mesh format
        let indices = validate_input_mesh(mesh)?;

        // Split the mesh into an initial list of meshlets (LOD 0)
        let vertex_buffer = mesh.get_vertex_buffer_data();
        let vertex_stride = mesh.get_vertex_size() as usize;
        let vertices = VertexDataAdapter::new(&vertex_buffer, vertex_stride, 0).unwrap();
        let mut meshlets = compute_meshlets(&indices, &vertices);
        let mut bounding_spheres = meshlets
            .iter()
            .map(|meshlet| compute_meshlet_bounds(meshlet, &vertices))
            .map(convert_meshlet_bounds)
            .map(|bounding_sphere| MeshletBoundingSpheres {
                self_culling: bounding_sphere,
                self_lod: MeshletBoundingSphere {
                    center: bounding_sphere.center,
                    radius: 0.0,
                },
                parent_lod: MeshletBoundingSphere {
                    center: bounding_sphere.center,
                    radius: f32::MAX,
                },
            })
            .collect::<Vec<_>>();
        let worst_case_meshlet_triangles = meshlets
            .meshlets
            .iter()
            .map(|m| m.triangle_count as u64)
            .sum();

        // Convert meshopt_Meshlet data to a custom format

        // Build further LODs
        let mut simplification_queue = 0..meshlets.len();
        let mut lod_level = 1;
        while simplification_queue.len() > 1 {
            // For each meshlet build a set of triangle edges
            let triangle_edges_per_meshlet =
                collect_triangle_edges_per_meshlet(simplification_queue.clone(), &meshlets);

            // For each meshlet build a list of connected meshlets (meshlets that share a triangle edge)
            let connected_meshlets_per_meshlet =
                find_connected_meshlets(simplification_queue.clone(), &triangle_edges_per_meshlet);

            // Group meshlets into roughly groups of 4, grouping meshlets with a high number of shared edges
            // http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
            let groups = group_meshlets(
                simplification_queue.clone(),
                &connected_meshlets_per_meshlet,
            );

            let next_lod_start = meshlets.len();

            for group_meshlets in groups.values().filter(|group| group.len() > 1) {
                // Simplify the group to ~50% triangle count
                let Some((simplified_group_indices, mut group_error)) =
                    simplify_meshlet_groups(group_meshlets, &meshlets, &vertices, lod_level)
                else {
                    continue;
                };

                // Add the maximum child error to the parent error to make parent error cumulative from LOD 0
                // (we're currently building the parent from its children)
                group_error += group_meshlets.iter().fold(group_error, |acc, meshlet_id| {
                    acc.max(bounding_spheres[*meshlet_id].self_lod.radius)
                });

                // Build a new LOD bounding sphere for the simplified group as a whole
                let mut group_bounding_sphere = convert_meshlet_bounds(compute_cluster_bounds(
                    &simplified_group_indices,
                    &vertices,
                ));
                group_bounding_sphere.radius = group_error;

                // For each meshlet in the group set their parent LOD bounding sphere to that of the simplified group
                for meshlet_id in group_meshlets {
                    bounding_spheres[*meshlet_id].parent_lod = group_bounding_sphere;
                }

                // Build new meshlets using the simplified group
                let new_meshlets_count = split_simplified_groups_into_new_meshlets(
                    &simplified_group_indices,
                    &vertices,
                    &mut meshlets,
                );

                // Calculate the culling bounding sphere for the new meshlets and set their LOD bounding spheres
                let new_meshlet_ids = (meshlets.len() - new_meshlets_count)..meshlets.len();
                bounding_spheres.extend(
                    new_meshlet_ids
                        .map(|meshlet_id| {
                            compute_meshlet_bounds(meshlets.get(meshlet_id), &vertices)
                        })
                        .map(convert_meshlet_bounds)
                        .map(|bounding_sphere| MeshletBoundingSpheres {
                            self_culling: bounding_sphere,
                            self_lod: group_bounding_sphere,
                            parent_lod: MeshletBoundingSphere {
                                center: group_bounding_sphere.center,
                                radius: f32::MAX,
                            },
                        }),
                );
            }

            simplification_queue = next_lod_start..meshlets.len();
            lod_level += 1;
        }

        let mut vertex_positions: Vec<EncodedVertexPosition>;
        #[allow(unsafe_code)]
        unsafe {
            let vertex_count = vertex_buffer.len() / vertex_stride;
            const POSITION_BYTE_OFFSET: usize = 0;
            let start = vertex_buffer.as_ptr().byte_add(POSITION_BYTE_OFFSET);
            vertex_positions = vec![EncodedVertexPosition::ZERO; vertex_count];
            for i in 0..vertex_count {
                let p: [f32; 3] = *(start.byte_add(i * vertex_stride) as *const [f32; 3]);
                let p_norm: [f32; 3] = core::array::from_fn(|i| p[i] / denorm_scale[i]);
                let p_quant = EncodedVertexPosition::from_f32(&p_norm);
                vertex_positions[i] = p_quant;
            }
        }

        Ok(Self {
            metadata: ModelMetadata {
                meshes_len: 1,
                meshlets_len: meshlets.len() as _,
                meshlet_indices_len: meshlets.triangles.len() as _,
                meshlet_vertices_len: meshlets.vertices.len() as _,
                vertices_len: vertex_positions.len() as _,
            },
            worst_case_meshlet_triangles,
            denorm_scale,
            vertex_data: vertex_buffer.into(),
            vertex_ids: meshlets.vertices.into(),
            indices: meshlets.triangles.into(),
            vertex_positions: vertex_positions.into(),
            meshlets: meshlets
                .meshlets
                .iter()
                .map(|m| Meshlet {
                    start_vertex_id: m.vertex_offset,
                    start_index_id: m.triangle_offset,
                    triangle_count: m.triangle_count,
                })
                .collect(),
            bounding_spheres: bounding_spheres.into(),
        })
    }

    pub fn export_for_xmetal(&self) {
        // pub type MeshletIndexType = u8;
        // pub type MeshletToMeshType = u8;
        // pub type MeshletVertexIndexType = u32;

        #[allow(non_camel_case_types)]
        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct packed_float3 {
            pub xyz: [f32; 3usize],
        }

        #[allow(non_camel_case_types)]
        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct packed_half3 {
            pub xyz: [half::f16; 3],
        }

        impl packed_half3 {
            pub const ZERO: Self = Self {
                xyz: [half::f16::from_f32_const(0.); 3],
            };
        }
        impl From<Vec3> for packed_half3 {
            fn from(value: Vec3) -> Self {
                packed_half3 {
                    xyz: [0, 1, 2].map(|c| half::f16::from_f32_const(value[c])),
                }
            }
        }

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct EncodedVertexPosition([u16; 3]);

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct Mesh {
            pub base_material_id: ::std::os::raw::c_uchar,
            pub roughness_material_id: ::std::os::raw::c_uchar,
            pub metalness_material_id: ::std::os::raw::c_uchar,
            pub normal_material_id: ::std::os::raw::c_uchar,
        }

        pub const MESHLET_BOUNDS__ROOT_LOD_ID: u16 = 65535;

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct MeshletBounds {
            pub lod_self_id: u16,
            pub lod_parent_id: u16,
            pub cone_apex: packed_half3,
            pub cone_axis: packed_half3,
            pub cone_cutoff: half::f16,
            pub center: packed_half3,
            pub radius: half::f16,
        }

        pub const MESHLET_TRIANGLE_COUNT_NUM_BITS: ::std::os::raw::c_uint = 8;
        pub const MESHLET_INDICES_NUM_BITS: ::std::os::raw::c_uint = 24;
        pub const MESHLET_INDEX_ALIGNMENT_POW_2: ::std::os::raw::c_uint = 2;
        const MESHLET_INDEX_ALIGNMENT: u32 = 1 << MESHLET_INDEX_ALIGNMENT_POW_2;
        const MESHLET_INDEX_ALIGNMENT_MASK: u32 = MESHLET_INDEX_ALIGNMENT - 1;

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq)]
        pub struct Meshlet {
            pub raw: u32,
            pub _vertices_start: u32,
        }
        impl Meshlet {
            #[inline]
            pub(crate) fn new(
                indices_start: u32,
                triangle_count: u32,
                vertices_start: u32,
            ) -> Self {
                assert!(
                    indices_start
                        < ((1 << MESHLET_INDICES_NUM_BITS) << MESHLET_INDEX_ALIGNMENT_POW_2),
                    "indices_start={indices_start} triangle_count={triangle_count}"
                );
                assert_eq!(
                    indices_start & MESHLET_INDEX_ALIGNMENT_MASK,
                    0,
                    "indices_start={indices_start} triangle_count={triangle_count}"
                );
                assert!(
                    triangle_count > 0
                        && (triangle_count - 1) < (1 << MESHLET_TRIANGLE_COUNT_NUM_BITS),
                    "indices_start={indices_start} triangle_count={triangle_count}"
                );
                Self {
                    raw: ((indices_start >> MESHLET_INDEX_ALIGNMENT_POW_2)
                        << MESHLET_TRIANGLE_COUNT_NUM_BITS)
                        | ((triangle_count - 1) as u32),
                    _vertices_start: vertices_start,
                }
            }

            // #[cfg(debug_assertions)]
            // pub fn indices_start(&self) -> u32 {
            //     (self.raw >> MESHLET_TRIANGLE_COUNT_NUM_BITS) << MESHLET_INDEX_ALIGNMENT_POW_2
            // }
            // #[cfg(debug_assertions)]
            // pub fn vertices_start(&self) -> u32 {
            //     self._vertices_start
            // }
            // #[cfg(debug_assertions)]
            // pub fn triangle_count(&self) -> u32 {
            //     (self.raw & ((1 << MESHLET_TRIANGLE_COUNT_NUM_BITS) - 1)) + 1
            // }
            // // TODO(1): Move helper into a more relevant place, like a new MeshIndices(Vec<MeshletIndexType>).
            // #[inline(always)]
            // pub(crate) fn indices_padding(index_count: u32) -> u32 {
            //     MESHLET_INDEX_ALIGNMENT_MASK
            //         & (MESHLET_INDEX_ALIGNMENT - (MESHLET_INDEX_ALIGNMENT_MASK & index_count))
            // }
        }

        fn write_file<T: Copy + Clone>(path: impl AsRef<Path>, t: &Arc<[T]>) {
            println!("- writing: {:?}", path.as_ref());
            #[allow(unsafe_code)]
            let a = unsafe {
                core::slice::from_raw_parts(
                    t.as_ptr() as *const _,
                    core::mem::size_of::<T>() * t.len(),
                )
            };
            std::fs::write(path, a).unwrap();
        }
        macro_rules! filepath {
            ($p:literal) => {
                concat!(
                    "/Users/pwong/projects/x-metal2/assets/generated/models/bevy-bunny/",
                    $p
                )
            };
        }

        write_file(filepath!("geometry.info"), &[self.metadata].into());
        write_file(
            filepath!("meshlets.bin"),
            &self
                .meshlets
                .iter()
                .map(|m| Meshlet::new(m.start_index_id, m.triangle_count, m.start_vertex_id))
                .collect::<Vec<Meshlet>>()
                .into(),
        );
        write_file(
            filepath!("bounds.bin"),
            &self
                .bounding_spheres
                .iter()
                .map(|b| MeshletBounds {
                    // TODO(0): Temporarily identify the LOD Level 0 for basic testing in x-metal2
                    lod_self_id: if b.self_lod.radius < f32::EPSILON {
                        MESHLET_BOUNDS__ROOT_LOD_ID
                    } else {
                        0
                    },
                    lod_parent_id: MESHLET_BOUNDS__ROOT_LOD_ID,
                    center: b.self_culling.center.into(),
                    radius: half::f16::from_f32_const(b.self_culling.radius),
                    cone_apex: packed_half3::ZERO,
                    cone_axis: packed_half3::ZERO,
                    cone_cutoff: half::f16::from_f32_const(1.),
                })
                .collect::<Vec<MeshletBounds>>()
                .into(),
        );
        write_file(filepath!("m_index.bin"), &self.indices);
        write_file(filepath!("m_vertex.bin"), &self.vertex_ids);
        write_file(filepath!("vertex_p.bin"), &self.vertex_positions);
        write_file(filepath!("meshes_ds.bin"), &[self.denorm_scale].into());
        write_file(
            filepath!("m_mesh.bin"),
            &core::iter::repeat(0)
                .take(self.meshlets.len())
                .collect::<Vec<u8>>()
                .into(),
        );
        write_file(
            filepath!("meshes.bin"),
            &[Mesh {
                base_material_id: 0,
                roughness_material_id: 0,
                metalness_material_id: 0,
                normal_material_id: 0,
            }]
            .into(),
        );
        // TODO(0): Implement vertices
        {
            #[repr(C)]
            #[derive(Copy, Clone, PartialEq)]
            pub struct EncodedVertex {
                pub positionxy: ::std::os::raw::c_uint,
                pub positionz_n1: ::std::os::raw::c_uint,
                pub n0_tangent_mikkt: ::std::os::raw::c_uint,
                pub tx_coord: ::std::os::raw::c_uint,
            }
            let vertices: Arc<[EncodedVertex]> = core::iter::repeat(0)
                .take(self.vertex_positions.len())
                .map(|_| EncodedVertex {
                    positionxy: 0,
                    positionz_n1: 0,
                    n0_tangent_mikkt: 0,
                    tx_coord: 0,
                })
                .collect::<Vec<EncodedVertex>>()
                .into();
            write_file(filepath!("vertex.bin"), &vertices);
        }
    }
}

fn validate_input_mesh(mesh: &Mesh) -> Result<Cow<'_, [u32]>, MeshToMeshletMeshConversionError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(MeshToMeshletMeshConversionError::WrongMeshPrimitiveTopology);
    }

    if mesh.attributes().map(|(id, _)| id).ne([
        Mesh::ATTRIBUTE_POSITION.id,
        Mesh::ATTRIBUTE_NORMAL.id,
        Mesh::ATTRIBUTE_UV_0.id,
        Mesh::ATTRIBUTE_TANGENT.id,
    ]) {
        return Err(MeshToMeshletMeshConversionError::WrongMeshVertexAttributes);
    }

    match mesh.indices() {
        Some(Indices::U32(indices)) => Ok(Cow::Borrowed(indices.as_slice())),
        Some(Indices::U16(indices)) => Ok(indices.iter().map(|i| *i as u32).collect()),
        _ => Err(MeshToMeshletMeshConversionError::MeshMissingIndices),
    }
}

fn compute_meshlets(indices: &[u32], vertices: &VertexDataAdapter) -> Meshlets {
    let mut meshlets = build_meshlets(indices, vertices, 64, 96, 0.0);

    for meshlet in &mut meshlets.meshlets {
        #[allow(unsafe_code)]
        #[allow(clippy::undocumented_unsafe_blocks)]
        unsafe {
            meshopt_optimizeMeshlet(
                &mut meshlets.vertices[meshlet.vertex_offset as usize],
                &mut meshlets.triangles[meshlet.triangle_offset as usize],
                meshlet.triangle_count as usize,
                meshlet.vertex_count as usize,
            );
        }
    }

    meshlets
}

fn collect_triangle_edges_per_meshlet(
    simplification_queue: Range<usize>,
    meshlets: &Meshlets,
) -> HashMap<usize, HashSet<(u32, u32)>> {
    let mut triangle_edges_per_meshlet = HashMap::new();
    for meshlet_id in simplification_queue {
        let meshlet = meshlets.get(meshlet_id);
        let meshlet_triangle_edges = triangle_edges_per_meshlet
            .entry(meshlet_id)
            .or_insert(HashSet::new());
        for i in meshlet.triangles.chunks(3) {
            let v0 = meshlet.vertices[i[0] as usize];
            let v1 = meshlet.vertices[i[1] as usize];
            let v2 = meshlet.vertices[i[2] as usize];
            meshlet_triangle_edges.insert((v0.min(v1), v0.max(v1)));
            meshlet_triangle_edges.insert((v0.min(v2), v0.max(v2)));
            meshlet_triangle_edges.insert((v1.min(v2), v1.max(v2)));
        }
    }
    triangle_edges_per_meshlet
}

fn find_connected_meshlets(
    simplification_queue: Range<usize>,
    triangle_edges_per_meshlet: &HashMap<usize, HashSet<(u32, u32)>>,
) -> HashMap<usize, Vec<(usize, usize)>> {
    let mut connected_meshlets_per_meshlet = HashMap::new();
    for meshlet_id in simplification_queue.clone() {
        connected_meshlets_per_meshlet.insert(meshlet_id, Vec::new());
    }

    for (meshlet_id1, meshlet_id2) in simplification_queue.tuple_combinations() {
        let shared_edge_count = triangle_edges_per_meshlet[&meshlet_id1]
            .intersection(&triangle_edges_per_meshlet[&meshlet_id2])
            .count();
        if shared_edge_count != 0 {
            connected_meshlets_per_meshlet
                .get_mut(&meshlet_id1)
                .unwrap()
                .push((meshlet_id2, shared_edge_count));
            connected_meshlets_per_meshlet
                .get_mut(&meshlet_id2)
                .unwrap()
                .push((meshlet_id1, shared_edge_count));
        }
    }
    connected_meshlets_per_meshlet
}

fn group_meshlets(
    simplification_queue: Range<usize>,
    connected_meshlets_per_meshlet: &HashMap<usize, Vec<(usize, usize)>>,
) -> HashMap<i32, Vec<usize>> {
    let mut xadj = Vec::with_capacity(simplification_queue.len() + 1);
    let mut adjncy = Vec::new();
    let mut adjwgt = Vec::new();
    for meshlet_id in simplification_queue.clone() {
        xadj.push(adjncy.len() as i32);
        for (connected_meshlet_id, shared_edge_count) in
            connected_meshlets_per_meshlet[&meshlet_id].iter().copied()
        {
            adjncy.push((connected_meshlet_id - simplification_queue.start) as i32);
            adjwgt.push(shared_edge_count as i32);
        }
    }
    xadj.push(adjncy.len() as i32);

    let mut group_per_meshlet = vec![0; simplification_queue.len()];
    let partition_count = (simplification_queue.len().div_ceil(4)) as i32;
    Graph::new(1, partition_count, &xadj, &adjncy)
        .unwrap()
        .set_adjwgt(&adjwgt)
        .part_kway(&mut group_per_meshlet)
        .unwrap();

    let mut groups = HashMap::new();
    for (i, meshlet_group) in group_per_meshlet.into_iter().enumerate() {
        groups
            .entry(meshlet_group)
            .or_insert(Vec::new())
            .push(i + simplification_queue.start);
    }
    groups
}

fn simplify_meshlet_groups(
    group_meshlets: &[usize],
    meshlets: &Meshlets,
    vertices: &VertexDataAdapter<'_>,
    lod_level: u32,
) -> Option<(Vec<u32>, f32)> {
    // Build a new index buffer into the mesh vertex data by combining all meshlet data in the group
    let mut group_indices = Vec::new();
    for meshlet_id in group_meshlets {
        let meshlet = meshlets.get(*meshlet_id);
        for meshlet_index in meshlet.triangles {
            group_indices.push(meshlet.vertices[*meshlet_index as usize]);
        }
    }

    // Allow more deformation for high LOD levels (1% at LOD 1, 10% at LOD 20+)
    let t = (lod_level - 1) as f32 / 19.0;
    let target_error = 0.1 * t + 0.01 * (1.0 - t);

    // Simplify the group to ~50% triangle count
    // TODO: Use simplify_with_locks()
    let mut error = 0.0;
    let simplified_group_indices = simplify(
        &group_indices,
        vertices,
        group_indices.len() / 2,
        target_error,
        SimplifyOptions::LockBorder,
        Some(&mut error),
    );

    // Check if we were able to simplify to at least 65% triangle count
    if simplified_group_indices.len() as f32 / group_indices.len() as f32 > 0.65 {
        return None;
    }

    // Convert error to object-space and convert from diameter to radius
    error *= simplify_scale(vertices) * 0.5;

    Some((simplified_group_indices, error))
}

fn split_simplified_groups_into_new_meshlets(
    simplified_group_indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    meshlets: &mut Meshlets,
) -> usize {
    let simplified_meshlets = compute_meshlets(simplified_group_indices, vertices);
    let new_meshlets_count = simplified_meshlets.len();

    let vertex_offset = meshlets.vertices.len() as u32;
    let triangle_offset = meshlets.triangles.len() as u32;
    meshlets
        .vertices
        .extend_from_slice(&simplified_meshlets.vertices);
    meshlets
        .triangles
        .extend_from_slice(&simplified_meshlets.triangles);
    meshlets
        .meshlets
        .extend(simplified_meshlets.meshlets.into_iter().map(|mut meshlet| {
            meshlet.vertex_offset += vertex_offset;
            meshlet.triangle_offset += triangle_offset;
            meshlet
        }));

    new_meshlets_count
}

fn convert_meshlet_bounds(bounds: meshopt_Bounds) -> MeshletBoundingSphere {
    MeshletBoundingSphere {
        center: bounds.center.into(),
        radius: bounds.radius,
    }
}

/// An error produced by [`MeshletMesh::from_mesh`].
#[derive(thiserror::Error, Debug)]
pub enum MeshToMeshletMeshConversionError {
    #[error("Mesh primitive topology is not TriangleList")]
    WrongMeshPrimitiveTopology,
    #[error("Mesh attributes are not {{POSITION, NORMAL, UV_0, TANGENT}}")]
    WrongMeshVertexAttributes,
    #[error("Mesh has no indices")]
    MeshMissingIndices,
}
