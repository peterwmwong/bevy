use super::{EncodedVertexPosition, ModelMetadata};
use bevy_math::Vec3;
use bevy_render::{
    mesh::{Indices, Mesh},
    render_resource::PrimitiveTopology,
};
use bevy_utils::{HashMap, HashSet};
use itertools::Itertools;
use meshopt::{
    build_meshlets, compute_cluster_bounds, compute_meshlet_bounds,
    ffi::{meshopt_Bounds, meshopt_Meshlet, meshopt_optimizeMeshlet},
    simplify, simplify_scale, Meshlets, SimplifyOptions, VertexDataAdapter,
};
use metis::Graph;
use std::{borrow::Cow, ops::Range, path::Path, sync::Arc};

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
pub fn export_meshlets_to_xmetal(
    mesh: &Mesh,
    denorm_scale: [f32; 3],
) -> Result<(), MeshToMeshletMeshConversionError> {
    const POSITION_BYTE_OFFSET: usize = 0;
    // Validate mesh format
    let indices = validate_input_mesh(mesh)?;

    // Split the mesh into an initial list of meshlets (LOD 0)
    let vertex_buffer = mesh.get_vertex_buffer_data();
    let vertex_stride = mesh.get_vertex_size() as usize;
    let vertices =
        VertexDataAdapter::new(&vertex_buffer, vertex_stride, POSITION_BYTE_OFFSET).unwrap();
    let mut lod_meshlets = LODMeshlets::new(compute_meshlets(&indices, &vertices));

    // Convert meshopt_Meshlet data to a custom format

    // Build further LODs
    let mut lod_level = 1;
    let mut meshlets_to_simplify = 0..lod_meshlets.meshlets.len();
    while meshlets_to_simplify.len() > 1 {
        let groups = create_groups(meshlets_to_simplify.clone(), &lod_meshlets);
        let next_meshlets_to_simplify_start = lod_meshlets.meshlets.len();

        for group in groups.values().filter(|group| group.len() > 1) {
            // Simplify the group to ~50% triangle count
            let Some((group_vertex_indices, group_error)) =
                simplify_meshlet_groups(group, &lod_meshlets, &vertices, lod_level)
            else {
                continue;
            };

            // Build a new LOD bounding sphere for the simplified group as a whole
            let group_lod = Group {
                // TODO(0): Replace `compute_cluster_bounds`/`meshopt_computeClusterBounds` with a
                //          MUCH simpler and FASTER center calculation
                // - meshopt_computeClusterBounds does a TON of unused calculations (radius, cone axis, cone apex, cone cutoff, etc.)
                // - Radius is IGNORED!
                // - Additionally, it CANNOT be inlined as it's an FFI call!
                // - Write a small, inlineable function to do the "center of vertices" calculation
                center: compute_cluster_bounds(&group_vertex_indices, &vertices)
                    .center
                    .into(),
                // Add the maximum child error to the parent error to make parent error cumulative from LOD 0
                // (we're currently building the parent from its children)
                error: group_error
                        // TODO(0): Remove group_error from fold(), should be 0
                        // - I think this is a subtle bug... since we're already adding group_error above
                        // - I *think* this is trying to ensure higher LOD levels always have a 
                        //   greater error
                        // - Addding group_error (above), should do it no need to limit the minimum 
                        //   error being added to group_error... to group_error
                        + group.iter().fold(group_error, |acc, &meshlet_id| {
                            acc.max(lod_meshlets.meshlet_group_error(meshlet_id))
                        }),
            };

            let lod_id = lod_meshlets.add_group(
                group_lod,
                // Build new meshlets using the simplified group
                &compute_meshlets(&group_vertex_indices, &vertices),
            );

            // For each meshlet in the group set their parent LOD bounding sphere to that of the simplified group
            lod_meshlets.assign_parent_group(lod_id, &group);
        }

        meshlets_to_simplify = next_meshlets_to_simplify_start..lod_meshlets.meshlets.len();
        lod_level += 1;
    }

    fn write_file<T: Copy + Clone>(path: impl AsRef<Path>, t: &Arc<[T]>) {
        println!("- writing: {:?}", path.as_ref());
        #[allow(unsafe_code)]
        let a = unsafe {
            core::slice::from_raw_parts(t.as_ptr() as *const _, core::mem::size_of::<T>() * t.len())
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
    write_file(
        filepath!("meshes.bin"),
        &[MeshMaterials {
            base_material_id: 0,
            roughness_material_id: 0,
            metalness_material_id: 0,
            normal_material_id: 0,
        }]
        .into(),
    );
    write_file(
        filepath!("m_mesh.bin"),
        &core::iter::repeat(0)
            .take(lod_meshlets.meshlets.len())
            .collect::<Vec<u8>>()
            .into(),
    );
    write_file(
        filepath!("geometry.info"),
        &[lod_meshlets.model_metadata(&vertices)].into(),
    );
    write_file(
        filepath!("meshlets.bin"),
        &lod_meshlets.generate_xmetal_meshlets(),
    );
    write_file(
        filepath!("bounds.bin"),
        &lod_meshlets.generate_meshlet_bounds(&vertices),
    );
    write_file(
        filepath!("m_groups.bin"),
        &lod_meshlets.meshlet_to_group_ids.into(),
    );
    write_file(
        filepath!("m_index.bin"),
        &lod_meshlets.meshlets.triangles.into(),
    );
    write_file(
        filepath!("m_vertex.bin"),
        &lod_meshlets.meshlets.vertices.into(),
    );

    {
        let vertex_count = vertex_buffer.len() / vertex_stride;
        #[allow(unsafe_code)]
        let vertex_positions: Vec<EncodedVertexPosition> = unsafe {
            let start = vertex_buffer.as_ptr().byte_add(POSITION_BYTE_OFFSET);
            (0..vertex_count)
                .map(|i| {
                    let p: [f32; 3] = *(start.byte_add(i * vertex_stride) as *const [f32; 3]);
                    let p_norm: [f32; 3] = core::array::from_fn(|i| p[i] / denorm_scale[i]);
                    let p_quant = EncodedVertexPosition::from_f32(&p_norm);
                    p_quant
                })
                .collect()
        };
        write_file(filepath!("vertex_p.bin"), &vertex_positions.into());

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq, Default)]
        pub struct EncodedVertex {
            pub positionxy: ::std::os::raw::c_uint,
            pub positionz_n1: ::std::os::raw::c_uint,
            pub n0_tangent_mikkt: ::std::os::raw::c_uint,
            pub tx_coord: ::std::os::raw::c_uint,
        }
        let vertices: Arc<[EncodedVertex]> = (vec![EncodedVertex::default(); vertex_count]).into();
        write_file(filepath!("vertex.bin"), &vertices);
    };
    write_file(filepath!("meshes_ds.bin"), &[denorm_scale].into());
    Ok(())
}

// TODO(0): Optimize with normaliztion/quantization
#[derive(Copy, Clone, PartialEq, Default)]
struct Group {
    pub center: Vec3,
    pub error: f32,
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
struct packed_half3 {
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

impl From<[f32; 3]> for packed_half3 {
    fn from(value: [f32; 3]) -> Self {
        packed_half3 {
            xyz: value.map(|v| half::f16::from_f32_const(v)),
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
struct MeshMaterials {
    pub base_material_id: ::std::os::raw::c_uchar,
    pub roughness_material_id: ::std::os::raw::c_uchar,
    pub metalness_material_id: ::std::os::raw::c_uchar,
    pub normal_material_id: ::std::os::raw::c_uchar,
}

#[derive(Clone, Copy)]
struct MeshletGroups {
    pub group_id: u16,
    pub parent_group_id: u16,
}

impl MeshletGroups {
    pub const NO_GROUP_ID: u16 = u16::MAX;
}

struct LODMeshlets {
    meshlets: Meshlets,
    meshlet_to_group_ids: Vec<MeshletGroups>,
    groups: Vec<Group>,
}

impl LODMeshlets {
    fn new(lod_0_meshlets: Meshlets) -> Self {
        Self {
            groups: vec![],
            meshlet_to_group_ids: (0..lod_0_meshlets.meshlets.len())
                .into_iter()
                .map(|_| MeshletGroups {
                    parent_group_id: MeshletGroups::NO_GROUP_ID,
                    group_id: MeshletGroups::NO_GROUP_ID,
                })
                .collect(),
            meshlets: lod_0_meshlets,
        }
    }

    fn assign_parent_group(&mut self, parent_group_id: u16, meshlet_ids: &[usize]) {
        for &m in meshlet_ids {
            self.meshlet_to_group_ids[m].parent_group_id = parent_group_id;
        }
    }

    fn add_group(&mut self, group: Group, group_meshlets: &Meshlets) -> u16 {
        let vertex_offset = self.meshlets.vertices.len() as u32;
        let triangle_offset = self.meshlets.triangles.len() as u32;

        self.meshlets
            .vertices
            .extend_from_slice(&group_meshlets.vertices);
        self.meshlets
            .triangles
            .extend_from_slice(&group_meshlets.triangles);

        self.meshlets
            .meshlets
            .extend(group_meshlets.meshlets.iter().map(|m| meshopt_Meshlet {
                vertex_offset: m.vertex_offset + vertex_offset,
                triangle_offset: m.triangle_offset + triangle_offset,
                vertex_count: m.vertex_count,
                triangle_count: m.triangle_count,
            }));

        let group_id = self.groups.len();
        assert!(group_id < (u16::MAX as usize));
        let group_id = group_id as u16;
        self.groups.push(group);

        self.meshlet_to_group_ids.extend_from_slice(&vec![
            MeshletGroups {
                parent_group_id: MeshletGroups::NO_GROUP_ID,
                group_id,
            };
            group_meshlets.meshlets.len()
        ]);
        group_id
    }

    fn meshlet_group_error(&self, meshlet_id: usize) -> f32 {
        let self_lod = self.meshlet_to_group_ids[meshlet_id].group_id;
        if self_lod == MeshletGroups::NO_GROUP_ID {
            0.
        } else {
            self.groups[self_lod as usize].error
        }
    }

    fn generate_meshlet_bounds(&self, vertices: &VertexDataAdapter) -> Arc<[MeshletBounds]> {
        self.meshlets
            .iter()
            .map(|m| compute_meshlet_bounds(m, vertices).into())
            .collect::<Vec<MeshletBounds>>()
            .into()
    }

    fn generate_xmetal_meshlets(&self) -> Arc<[Meshlet]> {
        self.meshlets
            .meshlets
            .iter()
            .map(|m| Meshlet::new(m.triangle_offset, m.triangle_count, m.vertex_offset))
            .collect::<Vec<Meshlet>>()
            .into()
    }

    fn model_metadata(&self, vertices: &VertexDataAdapter) -> ModelMetadata {
        ModelMetadata {
            meshes_len: 1,
            meshlets_len: self.meshlets.len() as _,
            meshlet_indices_len: self.meshlets.triangles.len() as _,
            meshlet_vertices_len: self.meshlets.vertices.len() as _,
            vertices_len: vertices.vertex_count as _,
        }
    }
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
    pub(crate) fn new(indices_start: u32, triangle_count: u32, vertices_start: u32) -> Self {
        assert!(
            indices_start < ((1 << MESHLET_INDICES_NUM_BITS) << MESHLET_INDEX_ALIGNMENT_POW_2),
            "indices_start={indices_start} triangle_count={triangle_count}"
        );
        assert_eq!(
            indices_start & MESHLET_INDEX_ALIGNMENT_MASK,
            0,
            "indices_start={indices_start} triangle_count={triangle_count}"
        );
        assert!(
            triangle_count > 0 && (triangle_count - 1) < (1 << MESHLET_TRIANGLE_COUNT_NUM_BITS),
            "indices_start={indices_start} triangle_count={triangle_count}"
        );
        Self {
            raw: ((indices_start >> MESHLET_INDEX_ALIGNMENT_POW_2)
                << MESHLET_TRIANGLE_COUNT_NUM_BITS)
                | ((triangle_count - 1) as u32),
            _vertices_start: vertices_start,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
struct MeshletBounds {
    cone_apex: packed_half3,
    cone_axis: packed_half3,
    cone_cutoff: half::f16,
    center: packed_half3,
    radius: half::f16,
}

impl From<meshopt_Bounds> for MeshletBounds {
    fn from(b: meshopt_Bounds) -> Self {
        MeshletBounds {
            center: b.center.into(),
            radius: half::f16::from_f32_const(b.radius),
            cone_apex: packed_half3::ZERO,
            cone_axis: packed_half3::ZERO,
            cone_cutoff: half::f16::from_f32_const(1.),
        }
    }
}

fn create_groups(
    meshlets_to_simplify: Range<usize>,
    meshlets: &LODMeshlets,
) -> bevy_utils::hashbrown::HashMap<i32, Vec<usize>> {
    // For each meshlet build a set of triangle edges
    let triangle_edges_per_meshlet =
        collect_triangle_edges_per_meshlet(meshlets_to_simplify.clone(), meshlets);

    // For each meshlet build a list of connected meshlets (meshlets that share a triangle edge)
    let connected_meshlets_per_meshlet =
        find_connected_meshlets(meshlets_to_simplify.clone(), &triangle_edges_per_meshlet);

    // Group meshlets into roughly groups of 4, grouping meshlets with a high number of shared edges
    // http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
    let groups = group_meshlets(
        meshlets_to_simplify.clone(),
        &connected_meshlets_per_meshlet,
    );
    groups
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
    meshlets: &LODMeshlets,
) -> HashMap<usize, HashSet<(u32, u32)>> {
    let mut triangle_edges_per_meshlet = HashMap::new();
    for meshlet_id in simplification_queue {
        let meshlet = meshlets.meshlets.get(meshlet_id);
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
    meshlets: &LODMeshlets,
    vertices: &VertexDataAdapter<'_>,
    lod_level: u32,
) -> Option<(Vec<u32>, f32)> {
    // Build a new index buffer into the mesh vertex data by combining all meshlet data in the group
    let mut group_indices = Vec::new();
    for meshlet_id in group_meshlets {
        let meshlet = meshlets.meshlets.get(*meshlet_id);
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
