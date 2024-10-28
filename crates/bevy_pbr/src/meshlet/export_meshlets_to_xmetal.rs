use bevy_math::Vec3;
use bevy_render::{
    mesh::{Indices, Mesh, MeshVertexAttribute},
    render_resource::PrimitiveTopology,
};
use bevy_utils::HashMap;
use derive_more::derive::{Display, Error};
use itertools::Itertools;
use meshopt::{
    build_meshlets, compute_meshlet_bounds,
    ffi::{meshopt_Bounds, meshopt_Meshlet, meshopt_optimizeMeshlet},
    generate_vertex_remap_multi, simplify_scale, simplify_with_attributes_and_locks, Meshlets,
    SimplifyOptions, VertexDataAdapter, VertexStream,
};
use metis::Graph;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Arc,
};

const TARGET_MESHLETS_PER_GROUP: usize = 8;
// Reject groups that keep over 95% of their original triangles
const SIMPLIFICATION_FAILURE_PERCENTAGE: f32 = 0.95;

const EXPECTED_MESH_ATTRIBUTES: [MeshVertexAttribute; 3] = [
    Mesh::ATTRIBUTE_POSITION,
    Mesh::ATTRIBUTE_NORMAL,
    Mesh::ATTRIBUTE_UV_0,
];
const POSITION_BYTE_OFFSET: usize = 0;
const NORMAL_BYTE_OFFSET: usize = 12;

fn write_model(
    filepath: PathBuf,
    vertices: &VertexDataAdapter<'_>,
    denorm_scale: [f32; 3],
    lod_meshlets: LODMeshlets,
) {
    println!("max lod level= {}", lod_meshlets.max_lod_level());
    println!("num meshlets= {}", lod_meshlets.meshlets.len());
    println!("num lod groups= {}", lod_meshlets.groups.len());

    fn write_file<T: Copy + Clone>(path: impl AsRef<Path>, t: &Arc<[T]>) {
        println!("- writing: {:?}", path.as_ref());
        #[allow(unsafe_code)]
        let a = unsafe {
            core::slice::from_raw_parts(t.as_ptr() as *const _, size_of::<T>() * t.len())
        };
        std::fs::write(path, a).unwrap();
    }
    let _ = std::fs::create_dir(&filepath);
    write_file(
        filepath.join("meshes.bin"),
        &[MeshMaterials {
            base_material_id: 0,
            roughness_material_id: 0,
            metalness_material_id: 0,
            normal_material_id: 0,
        }]
        .into(),
    );
    write_file(
        filepath.join("m_mesh.bin"),
        &core::iter::repeat(0)
            .take(lod_meshlets.meshlets.len())
            .collect::<Vec<u8>>()
            .into(),
    );
    write_file(
        filepath.join("geometry.info"),
        &[lod_meshlets.model_metadata(&vertices)].into(),
    );
    write_file(
        filepath.join("meshlets.bin"),
        &lod_meshlets.generate_xmetal_meshlets(),
    );
    write_file(
        filepath.join("bounds.bin"),
        &lod_meshlets.generate_meshlet_bounds(&vertices),
    );
    write_file(
        filepath.join("m_groups.bin"),
        &lod_meshlets.meshlet_to_lod_groups.into(),
    );
    write_file(filepath.join("lod_groups.bin"), &lod_meshlets.groups.into());
    write_file(
        filepath.join("m_index.bin"),
        &lod_meshlets.meshlets.triangles.into(),
    );
    write_file(
        filepath.join("m_vertex.bin"),
        &lod_meshlets.meshlets.vertices.into(),
    );
    write_file(
        filepath.join("dbg_m_lods.bin"),
        &lod_meshlets.dbg_meshlet_to_lod_level.into(),
    );
    write_file(
        filepath.join("lod_m_end.bin"),
        &lod_meshlets.lod_level_to_meshlet_end.into(),
    );
    write_file(
        filepath.join("g_lods.bin"),
        &lod_meshlets.group_to_lod_level.into(),
    );

    // TODO(0): Add generating meshes_sbr.bin see `tmp_generate_meshes_sphere_bounds_radius()` on
    // `select_meshlets-remove-threadgroup-mem` branch

    {
        let mut max_sphere_bounds_radius = f32::MIN;
        let vertex_count = vertices.vertex_count;
        #[allow(unsafe_code)]
        let vertex_positions: Vec<EncodedVertexPosition> = unsafe {
            let start = vertices.pos_ptr() as *const u8;
            (0..vertex_count)
                .map(|i| {
                    let p: [f32; 3] =
                        *(start.byte_add(i * vertices.vertex_stride) as *const [f32; 3]);
                    max_sphere_bounds_radius = max_sphere_bounds_radius
                        .max((p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt());
                    let p_norm: [f32; 3] = core::array::from_fn(|i| p[i] / denorm_scale[i]);
                    let p_quant = EncodedVertexPosition::from_f32(&p_norm);
                    p_quant
                })
                .collect()
        };
        write_file(filepath.join("vertex_p.bin"), &vertex_positions.into());

        #[repr(C)]
        #[derive(Copy, Clone, PartialEq, Default)]
        pub struct EncodedVertex {
            pub positionxy: ::std::os::raw::c_uint,
            pub positionz_n1: ::std::os::raw::c_uint,
            pub n0_tangent_mikkt: ::std::os::raw::c_uint,
            pub tx_coord: ::std::os::raw::c_uint,
        }
        let vertices: Arc<[EncodedVertex]> = (vec![EncodedVertex::default(); vertex_count]).into();
        write_file(filepath.join("vertex.bin"), &vertices);

        write_file(
            filepath.join("meshes_sbr.bin"),
            &[max_sphere_bounds_radius].into(),
        );
    };
    write_file(filepath.join("meshes_ds.bin"), &[denorm_scale].into());
}

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
    // Validate mesh format
    let indices = validate_input_mesh(mesh)?;

    // Split the mesh into an initial list of meshlets (LOD 0)
    let vertex_buffer = mesh.create_packed_vertex_buffer_data();
    let vertex_stride = mesh.get_vertex_size() as usize;
    let vertices =
        VertexDataAdapter::new(&vertex_buffer, vertex_stride, POSITION_BYTE_OFFSET).unwrap();
    let mut lod_levels: Vec<LODLevel> = vec![LODLevel::new(compute_meshlets(&indices, &vertices))];

    // Generate a position-only vertex buffer for determining what meshlets are connected for use in grouping
    let (position_only_vertex_count, position_only_vertex_remap) = generate_vertex_remap_multi(
        vertices.vertex_count,
        &[VertexStream::new_with_stride::<Vec3, _>(
            vertex_buffer.as_ptr(),
            vertex_stride,
        )],
        Some(&indices),
    );

    let mut vertex_locks = vec![false; vertices.vertex_count];

    loop {
        let child_level = lod_levels.last_mut().unwrap();
        let parent_groups = {
            let connected_meshlets_per_meshlet = find_connected_meshlets(
                &child_level.meshlets,
                &position_only_vertex_remap,
                position_only_vertex_count,
            );

            // Group meshlets into roughly groups of size TARGET_MESHLETS_PER_GROUP,
            // grouping meshlets with a high number of shared vertices
            let groups = group_meshlets(&connected_meshlets_per_meshlet);

            // Lock borders between groups to prevent cracks when simplifying
            lock_group_borders(
                &mut vertex_locks,
                &groups,
                &child_level.meshlets,
                &position_only_vertex_remap,
                position_only_vertex_count,
            );

            groups
        };
        let mut parent_level = LODLevel::empty();

        for parent_group in parent_groups.iter().filter(|group| group.len() > 1) {
            let vertex_normals: &[f32] = bytemuck::cast_slice(
                &vertex_buffer[NORMAL_BYTE_OFFSET..(NORMAL_BYTE_OFFSET + size_of::<f32>())],
            );
            // Simplify the group to ~50% triangle count
            let Some((parent_group_vertex_indices, parent_group_error)) = simplify_meshlet_group(
                parent_group,
                &child_level.meshlets,
                &vertices,
                vertex_normals,
                vertex_stride,
                &vertex_locks,
            ) else {
                // TODO(0): Consider a retry_queue
                continue;
            };

            // Build a new LOD bounding sphere for the simplified group as a whole
            let parent_group_id = parent_level.add_group(
                Group {
                    // TODO(0): Replace `compute_cluster_bounds`/`meshopt_computeClusterBounds` with a
                    //          MUCH simpler and FASTER center calculation
                    // - meshopt_computeClusterBounds does a TON of unused calculations (radius, cone axis, cone apex, cone cutoff, etc.)
                    // - Radius is IGNORED!
                    // - Additionally, it CANNOT be inlined as it's an FFI call!
                    // - Write a small, inlineable function to do the "center of vertices" calculation
                    center: compute_cluster_center(&parent_group_vertex_indices, &vertices).into(),
                    // Add the maximum child error to the parent error to make parent error cumulative from LOD 0
                    // (we're currently building the parent from its children)
                    // TODO(0): Remove `group_error +`, see bevy change: https://github.com/bevyengine/bevy/pull/15023
                    error: parent_group_error
                        + parent_group
                            .iter()
                            .fold(parent_group_error, |acc, &meshlet_id| {
                                acc.max(child_level.meshlet_group_error(meshlet_id))
                            }),
                },
                // Build new meshlets using the simplified group
                compute_meshlets(&parent_group_vertex_indices, &vertices),
            );

            // For each meshlet in the group set their parent LOD bounding sphere to that of the simplified group
            child_level.assign_parent_group(parent_group_id, &parent_group);
        }

        if parent_level.is_empty() {
            break;
        }
        lod_levels.push(parent_level);
    }

    let model_name = "bevy-bunny-upgrade-meshopt";
    println!("export_meshlets_to_metal: Writing ");
    write_model(
        format!("/Users/pwong/projects/x-metal2/assets/generated/models/{model_name}/").into(),
        &vertices,
        denorm_scale,
        LODMeshlets::ascending(lod_levels.iter()),
    );
    // write_model(
    //     "/Users/pwong/projects/x-metal2/assets/generated/models/bevy-bunny-DESCENDING/".into(),
    //     &vertices,
    //     denorm_scale,
    //     LODMeshlets::descending(lod_levels.iter()),
    // );
    Ok(())
}

#[inline]
fn compute_cluster_center(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> [f32; 3] {
    let get_point = |id: usize| -> [f32; 3] {
        unsafe { *(vertices.pos_ptr().byte_add(id * vertices.vertex_stride) as *const [f32; 3]) }
    };

    let mut pmin = [(0_usize, f32::MAX); 3];
    let mut pmax = [(0_usize, f32::MIN); 3];
    for tri in indices.chunks_exact(3) {
        for &vid in tri {
            let vid = vid as usize;
            let p: [f32; 3] = get_point(vid);
            for axis in 0..3 {
                let v = p[axis];
                if v < pmin[axis].1 {
                    pmin[axis] = (vid, v);
                }
                if v > pmax[axis].1 {
                    pmax[axis] = (vid, v);
                }
            }
        }
    }

    let mut paxisd2 = 0.;
    let mut paxis = 0;
    for axis in 0..3 {
        let p1 = get_point(pmin[axis].0);
        let p2 = get_point(pmax[axis].0);
        let d2 = (p2[0] - p1[0]) * (p2[0] - p1[0])
            + (p2[1] - p1[1]) * (p2[1] - p1[1])
            + (p2[2] - p1[2]) * (p2[2] - p1[2]);

        if d2 > paxisd2 {
            paxisd2 = d2;
            paxis = axis;
        }
    }

    let p1 = get_point(pmin[paxis].0);
    let p2 = get_point(pmax[paxis].0);
    let mut center = [
        (p1[0] + p2[0]) / 2.,
        (p1[1] + p2[1]) / 2.,
        (p1[2] + p2[2]) / 2.,
    ];
    let mut radius = paxisd2.sqrt() / 2.;

    for tri in indices.chunks_exact(3) {
        for &vid in tri {
            let p = get_point(vid as usize);
            let d2 = (p[0] - center[0]) * (p[0] - center[0])
                + (p[1] - center[1]) * (p[1] - center[1])
                + (p[2] - center[2]) * (p[2] - center[2]);

            if d2 > radius * radius {
                let d = d2.sqrt();
                assert!(d > 0.);

                let k = 0.5 + (radius / d) / 2.;
                center[0] = center[0] * k + p[0] * (1. - k);
                center[1] = center[1] * k + p[1] * (1. - k);
                center[2] = center[2] * k + p[2] * (1. - k);
                radius = (radius + d) / 2.;
            }
        }
    }
    center
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodedVertexPosition([i16; 3]);
impl EncodedVertexPosition {
    #[inline]
    pub fn from_f32(&p: &[f32; 3]) -> Self {
        assert!(
            p.iter().find(|&&c| c < -1. || c > 1.).is_none(),
            "BAD POSITION: {p:?}"
        );
        Self(p.map(|c| (c * (i16::MAX as f32)) as i16))
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ModelMetadata {
    pub(crate) meshes_len: u32,
    pub(crate) lod_groups_len: u32,
    pub(crate) lod_levels_len: u32,
    pub(crate) meshlets_len: u32,
    // TODO(0): Replace indices_len with triangle_count to save 1.5 bits (or increase
    //          amount by 3x).
    pub(crate) meshlet_indices_len: u32,
    pub(crate) meshlet_vertices_len: u32,
    pub(crate) vertices_len: u32,
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

struct XMetal2Meshlets {
    pub meshlets: Vec<meshopt_Meshlet>,
    pub vertices: Vec<u32>,
    pub triangles: Vec<[u8; 3]>,
}

impl XMetal2Meshlets {
    fn empty() -> Self {
        Self {
            meshlets: vec![],
            vertices: vec![],
            triangles: vec![],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.meshlets.len()
    }

    fn meshlet_from_ffi<'a>(&'a self, meshlet: &meshopt_Meshlet) -> meshopt::Meshlet<'a> {
        #[allow(unsafe_code)]
        let triangles: &'a [u8] = unsafe {
            core::slice::from_raw_parts(
                self.triangles[meshlet.triangle_offset as usize].as_ptr(),
                (meshlet.triangle_count as usize) * 3,
            )
        };
        meshopt::Meshlet {
            vertices: &self.vertices[meshlet.vertex_offset as usize
                ..meshlet.vertex_offset as usize + meshlet.vertex_count as usize],
            triangles,
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> meshopt::Meshlet<'_> {
        self.meshlet_from_ffi(&self.meshlets[idx])
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = meshopt::Meshlet<'_>> {
        self.meshlets
            .iter()
            .map(|meshlet| self.meshlet_from_ffi(meshlet))
    }
}

impl From<Meshlets> for XMetal2Meshlets {
    fn from(mut value: Meshlets) -> Self {
        let mut triangles: Vec<[u8; 3]> = Vec::new();
        for m in &mut value.meshlets {
            let new_triangle_offset = triangles.len();
            for t in 0..(m.triangle_count as usize) {
                triangles.push([
                    value.triangles[(m.triangle_offset as usize) + (t * 3) + 0],
                    value.triangles[(m.triangle_offset as usize) + (t * 3) + 1],
                    value.triangles[(m.triangle_offset as usize) + (t * 3) + 2],
                ]);
            }
            m.triangle_offset = new_triangle_offset as _;
        }
        Self {
            meshlets: value.meshlets,
            vertices: value.vertices,
            triangles,
        }
    }
}

struct LODLevel {
    pub meshlets: XMetal2Meshlets,
    meshlet_to_lod_groups: Vec<MeshletGroups>,
    groups: Vec<Group>,
}

impl LODLevel {
    fn new(meshlets: XMetal2Meshlets) -> Self {
        Self {
            groups: vec![],
            meshlet_to_lod_groups: meshlets
                .meshlets
                .iter()
                .map(|_| MeshletGroups {
                    parent_group_id: MeshletGroups::NO_GROUP_ID,
                    group_id: MeshletGroups::NO_GROUP_ID,
                })
                .collect(),
            meshlets,
        }
    }

    fn empty() -> Self {
        Self {
            groups: vec![],
            meshlet_to_lod_groups: vec![],
            meshlets: XMetal2Meshlets::empty(),
        }
    }

    fn is_empty(&self) -> bool {
        self.groups.len() == 0
    }

    fn assign_parent_group(&mut self, parent_group_id: u16, meshlet_ids: &[usize]) {
        for &m in meshlet_ids {
            self.meshlet_to_lod_groups[m].parent_group_id = parent_group_id;
        }
    }

    fn add_group(&mut self, group: Group, group_meshlets: XMetal2Meshlets) -> u16 {
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

        self.meshlet_to_lod_groups.extend_from_slice(&vec![
            MeshletGroups {
                parent_group_id: MeshletGroups::NO_GROUP_ID,
                group_id,
            };
            group_meshlets.meshlets.len()
        ]);
        group_id
    }

    fn meshlet_group_error(&self, meshlet_id: usize) -> f32 {
        let self_lod = self.meshlet_to_lod_groups[meshlet_id].group_id;
        if self_lod == MeshletGroups::NO_GROUP_ID {
            0.
        } else {
            self.groups[self_lod as usize].error
        }
    }
}

struct LODMeshlets {
    pub meshlets: XMetal2Meshlets,
    group_to_lod_level: Vec<u8>,
    lod_level_to_meshlet_end: Vec<u16>,
    meshlet_to_lod_groups: Vec<MeshletGroups>,
    groups: Vec<Group>,
    dbg_meshlet_to_lod_level: Vec<u8>,
}

struct LodLevelIterator<'a, T: ExactSizeIterator<Item = (usize, &'a LODLevel)>> {
    lod_levels: T,
    is_descending_levels_of_detail: bool,
}

impl LODMeshlets {
    #[allow(unused)]
    fn descending<'a>(descending_lod_levels: impl ExactSizeIterator<Item = &'a LODLevel>) -> Self {
        Self::new(LodLevelIterator {
            lod_levels: descending_lod_levels.enumerate(),
            is_descending_levels_of_detail: true,
        })
    }
    fn ascending<'a>(
        descending_lod_levels: impl ExactSizeIterator<Item = &'a LODLevel> + DoubleEndedIterator,
    ) -> Self {
        Self::new(LodLevelIterator {
            lod_levels: descending_lod_levels.enumerate().rev(),
            is_descending_levels_of_detail: false,
        })
    }

    fn new<'a>(
        LodLevelIterator {
            lod_levels,
            is_descending_levels_of_detail,
        }: LodLevelIterator<'a, impl ExactSizeIterator<Item = (usize, &'a LODLevel)>>,
    ) -> Self {
        let mut meshlets = XMetal2Meshlets::empty();
        let mut meshlet_to_lod_groups: Vec<MeshletGroups> = vec![];
        let mut groups: Vec<Group> = vec![];
        let mut group_to_lod_level: Vec<u8> = vec![];
        let mut lod_level_to_meshlet_end: Vec<u16> = vec![];
        let mut dbg_meshlet_to_lod_level: Vec<u8> = vec![(lod_levels.len() - 1) as u8];

        let mut prev_level_groups_offset = 0;
        println!("[LODMeshlets] LOD Levels");
        for (actual_lod_level, (lod_level, level)) in lod_levels.enumerate() {
            let triangle_offset: u32 = meshlets.triangles.len() as _;
            let vertices_offset: u32 = meshlets.vertices.len() as _;

            meshlets
                .triangles
                .extend_from_slice(&level.meshlets.triangles);
            meshlets
                .vertices
                .extend_from_slice(&level.meshlets.vertices);
            let num_meshlets = level.meshlets.meshlets.len();
            meshlets
                .meshlets
                .extend(level.meshlets.meshlets.iter().map(|m| meshopt_Meshlet {
                    triangle_offset: triangle_offset + m.triangle_offset,
                    vertex_offset: vertices_offset + m.vertex_offset,
                    vertex_count: m.vertex_count,
                    triangle_count: m.triangle_count,
                }));

            lod_level_to_meshlet_end.push(meshlets.meshlets.len() as _);

            let cur_level_offset: u16 = groups.len() as _;
            let num_groups = level.groups.len();
            let next_level_offset: u16 = cur_level_offset + (num_groups as u16);
            groups.extend(level.groups.iter().map(|g| Group {
                center: g.center,
                // IMPORTANT: Optimization Precalculate - Runtime LOD selection only uses the error squared.
                // - See lod_group.h projected_sphere_area
                error: g.error * g.error,
            }));
            group_to_lod_level.extend(core::iter::repeat_n(actual_lod_level as u8, num_groups));

            println!(
                "    [{actual_lod_level}] lod_groups: {num_groups} meshlets: {num_meshlets} lod_level_to_meshlet_end={}",
                *lod_level_to_meshlet_end.last().unwrap()
            );

            meshlet_to_lod_groups.extend(level.meshlet_to_lod_groups.iter().map(|g| {
                MeshletGroups {
                    group_id: cur_level_offset + g.group_id,
                    parent_group_id: if is_descending_levels_of_detail {
                        next_level_offset + g.parent_group_id
                    } else {
                        prev_level_groups_offset + g.parent_group_id
                    },
                }
            }));

            prev_level_groups_offset = cur_level_offset;

            dbg_meshlet_to_lod_level.extend(core::iter::repeat_n(lod_level as u8, num_meshlets));
        }

        Self {
            dbg_meshlet_to_lod_level,
            group_to_lod_level,
            groups,
            lod_level_to_meshlet_end,
            meshlet_to_lod_groups,
            meshlets,
        }
    }

    fn max_lod_level(&self) -> u8 {
        self.dbg_meshlet_to_lod_level[0]
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
            .map(|m| {
                Meshlet::new(
                    m.triangle_offset,
                    m.triangle_count,
                    m.vertex_offset,
                    m.vertex_count,
                )
            })
            .collect::<Vec<Meshlet>>()
            .into()
    }

    fn model_metadata(&self, vertices: &VertexDataAdapter) -> ModelMetadata {
        ModelMetadata {
            meshes_len: 1,
            lod_groups_len: self.groups.len() as _,
            meshlets_len: self.meshlets.len() as _,
            meshlet_indices_len: self.meshlets.triangles.len() as _,
            meshlet_vertices_len: self.meshlets.vertices.len() as _,
            vertices_len: vertices.vertex_count as _,
            lod_levels_len: self.lod_level_to_meshlet_end.len() as _,
        }
    }
}

pub const MESHLET_TRIANGLE_COUNT_NUM_BITS: ::std::os::raw::c_uint = 8;
pub const MESHLET_INDICES_NUM_BITS: ::std::os::raw::c_uint = 24;
pub const MESHLET_VERTEX_COUNT_NUM_BITS: ::std::os::raw::c_uint = 8;
pub const MESHLET_VERTEX_START_NUM_BITS: ::std::os::raw::c_uint = 24;

const MESHLET_VERTEX_COUNT_MASK: u32 = (1 << MESHLET_VERTEX_COUNT_NUM_BITS) - 1;
const MESHLET_TRIANGLE_COUNT_MASK: u32 = (1 << MESHLET_TRIANGLE_COUNT_NUM_BITS) - 1;

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub struct Meshlet {
    pub raw: u32,
    pub _vertices: u32,
}

impl Meshlet {
    #[inline]
    pub(crate) fn new(
        indices_start: u32,
        triangle_count: u32,
        vertices_start: u32,
        vertices_count: u32,
    ) -> Self {
        const MAX_INDICES_START: u32 = (1 << MESHLET_INDICES_NUM_BITS) - 1;
        assert!(
            indices_start <= MAX_INDICES_START,
            "indices_start ({indices_start}) must be <= {MAX_INDICES_START}"
        );
        assert!(
            (1..=MESHLET_VERTEX_COUNT_MASK).contains(&triangle_count),
            "triangle_count ({triangle_count}) must be within 1..={MESHLET_VERTEX_COUNT_MASK}"
        );
        const MAX_VERTICES_START: u32 = (1 << MESHLET_VERTEX_START_NUM_BITS) - 1;
        assert!(
            vertices_start <= MAX_VERTICES_START,
            "vertices_start ({vertices_start}) must be <= {MAX_VERTICES_START}"
        );
        assert!(
            (1..=MESHLET_TRIANGLE_COUNT_MASK).contains(&vertices_count),
            "vertices_count ({vertices_count}) must be within 1..={MESHLET_TRIANGLE_COUNT_MASK}"
        );
        Self {
            raw: ((indices_start << MESHLET_TRIANGLE_COUNT_NUM_BITS) | triangle_count),
            _vertices: ((vertices_start << MESHLET_VERTEX_COUNT_NUM_BITS) | vertices_count),
        }
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    pub fn indices_start(&self) -> u32 {
        self.raw >> MESHLET_TRIANGLE_COUNT_NUM_BITS
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    pub fn vertices_start(&self) -> u32 {
        self._vertices >> MESHLET_VERTEX_COUNT_NUM_BITS
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    pub fn vertices_count(&self) -> u32 {
        self._vertices & MESHLET_VERTEX_COUNT_MASK
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    pub fn triangle_count(&self) -> u32 {
        self.raw & ((1 << MESHLET_TRIANGLE_COUNT_NUM_BITS) - 1)
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
            cone_apex: b.cone_apex.into(),
            cone_axis: b.cone_axis.into(),
            cone_cutoff: half::f16::from_f32_const(b.cone_cutoff),
            radius: half::f16::from_f32_const(b.radius),
        }
    }
}

fn validate_input_mesh(mesh: &Mesh) -> Result<Cow<'_, [u32]>, MeshToMeshletMeshConversionError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(MeshToMeshletMeshConversionError::WrongMeshPrimitiveTopology);
    }

    // TODO(1): Change these asserts to some form of const assert.
    assert_eq!(EXPECTED_MESH_ATTRIBUTES[0].id, Mesh::ATTRIBUTE_POSITION.id);
    assert_eq!(EXPECTED_MESH_ATTRIBUTES[1].id, Mesh::ATTRIBUTE_NORMAL.id);
    assert_eq!(
        NORMAL_BYTE_OFFSET,
        Mesh::ATTRIBUTE_POSITION.format.size() as usize
    );
    if mesh
        .attributes()
        .map(|(attribute, _)| attribute.id)
        .ne(EXPECTED_MESH_ATTRIBUTES.map(|i| i.id))
    {
        return Err(MeshToMeshletMeshConversionError::WrongMeshVertexAttributes);
    }

    match mesh.indices() {
        Some(Indices::U32(indices)) => Ok(Cow::Borrowed(indices.as_slice())),
        Some(Indices::U16(indices)) => Ok(indices.iter().map(|i| *i as u32).collect()),
        _ => Err(MeshToMeshletMeshConversionError::MeshMissingIndices),
    }
}

fn compute_meshlets(indices: &[u32], vertices: &VertexDataAdapter) -> XMetal2Meshlets {
    // let mut meshlets = build_meshlets(indices, vertices, 64, 96, 0.0);
    // let mut meshlets = build_meshlets(indices, vertices, 64, 128, 0.0);
    // let mut meshlets = build_meshlets(indices, vertices, 64, 192, 0.0);
    // let mut meshlets = build_meshlets(indices, vertices, 128, 192, 0.0);
    // let mut meshlets = build_meshlets(indices, vertices, 32, 32, 0.0);
    let mut meshlets = build_meshlets(indices, vertices, 160, 192, 0.0);
    let last_meshlet = &meshlets.meshlets[meshlets.meshlets.len() - 1];
    meshlets
        .vertices
        .truncate((last_meshlet.vertex_offset + last_meshlet.vertex_count) as usize);
    meshlets
        .triangles
        .truncate((last_meshlet.triangle_offset + last_meshlet.triangle_count * 3) as usize);

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

    meshlets.into()
}

fn find_connected_meshlets(
    meshlets: &XMetal2Meshlets,
    position_only_vertex_remap: &[u32],
    position_only_vertex_count: usize,
) -> Vec<Vec<(usize, usize)>> {
    // For each vertex, build a list of all meshlets that use it
    let mut vertices_to_meshlets = vec![Vec::new(); position_only_vertex_count];
    for (meshlet_queue_id, meshlet) in meshlets.iter().enumerate() {
        for index in meshlet.triangles {
            let vertex_id = position_only_vertex_remap[meshlet.vertices[*index as usize] as usize];
            let vertex_to_meshlets = &mut vertices_to_meshlets[vertex_id as usize];
            // Meshlets are added in order, so we can just check the last element to deduplicate,
            // in the case of two triangles sharing the same vertex within a single meshlet
            if vertex_to_meshlets.last() != Some(&meshlet_queue_id) {
                vertex_to_meshlets.push(meshlet_queue_id);
            }
        }
    }

    // For each meshlet pair, count how many vertices they share
    let mut meshlet_pair_to_shared_vertex_count = HashMap::new();
    for vertex_meshlet_ids in vertices_to_meshlets {
        for (meshlet_queue_id1, meshlet_queue_id2) in
            vertex_meshlet_ids.into_iter().tuple_combinations()
        {
            let count = meshlet_pair_to_shared_vertex_count
                .entry((
                    meshlet_queue_id1.min(meshlet_queue_id2),
                    meshlet_queue_id1.max(meshlet_queue_id2),
                ))
                .or_insert(0);
            *count += 1;
        }
    }

    // For each meshlet, gather all other meshlets that share at least one vertex along with their shared vertex count
    let mut connected_meshlets_per_meshlet = vec![Vec::new(); meshlets.len()];
    for ((meshlet_queue_id1, meshlet_queue_id2), shared_count) in
        meshlet_pair_to_shared_vertex_count
    {
        // We record both id1->id2 and id2->id1 as adjacency is symmetrical
        connected_meshlets_per_meshlet[meshlet_queue_id1].push((meshlet_queue_id2, shared_count));
        connected_meshlets_per_meshlet[meshlet_queue_id2].push((meshlet_queue_id1, shared_count));
    }

    // The order of meshlets depends on hash traversal order; to produce deterministic results, sort them
    for list in connected_meshlets_per_meshlet.iter_mut() {
        // TODO(0): reconsider stable sort for more deterministic results
        list.sort_unstable();
    }

    connected_meshlets_per_meshlet
}

// METIS manual: https://github.com/KarypisLab/METIS/blob/e0f1b88b8efcb24ffa0ec55eabb78fbe61e58ae7/manual/manual.pdf
fn group_meshlets(
    connected_meshlets_per_meshlet: &[Vec<(usize, usize)>],
) -> Vec<SmallVec<[usize; TARGET_MESHLETS_PER_GROUP]>> {
    let mut xadj = Vec::with_capacity(connected_meshlets_per_meshlet.len() + 1);
    let mut adjncy = Vec::new();
    let mut adjwgt = Vec::new();
    for connected_meshlets in connected_meshlets_per_meshlet {
        xadj.push(adjncy.len() as i32);
        for (connected_meshlet_queue_id, shared_vertex_count) in connected_meshlets {
            adjncy.push(*connected_meshlet_queue_id as i32);
            adjwgt.push(*shared_vertex_count as i32);
            // TODO: Additional weight based on meshlet spatial proximity
        }
    }
    xadj.push(adjncy.len() as i32);

    let mut group_per_meshlet = vec![0; connected_meshlets_per_meshlet.len()];
    let partition_count = connected_meshlets_per_meshlet
        .len()
        .div_ceil(TARGET_MESHLETS_PER_GROUP); // TODO: Nanite uses groups of 8-32, probably based on some kind of heuristic
    Graph::new(1, partition_count as i32, &xadj, &adjncy)
        .unwrap()
        .set_option(metis::option::Seed(17))
        .set_adjwgt(&adjwgt)
        .part_kway(&mut group_per_meshlet)
        .unwrap();

    let mut groups = vec![SmallVec::new(); partition_count];
    for (meshlet_queue_id, meshlet_group) in group_per_meshlet.into_iter().enumerate() {
        groups[meshlet_group as usize].push(meshlet_queue_id);
    }
    groups
}

fn lock_group_borders(
    vertex_locks: &mut [bool],
    groups: &[SmallVec<[usize; TARGET_MESHLETS_PER_GROUP]>],
    meshlets: &XMetal2Meshlets,
    position_only_vertex_remap: &[u32],
    position_only_vertex_count: usize,
) {
    let mut position_only_locks = vec![-1; position_only_vertex_count];

    // Iterate over position-only based vertices of all meshlets in all groups
    for (group_id, group_meshlets) in groups.iter().enumerate() {
        for meshlet_id in group_meshlets {
            let meshlet = meshlets.get(*meshlet_id);
            for index in meshlet.triangles {
                let vertex_id =
                    position_only_vertex_remap[meshlet.vertices[*index as usize] as usize] as usize;

                // If the vertex is not yet claimed by any group, or was already claimed by this group
                if position_only_locks[vertex_id] == -1
                    || position_only_locks[vertex_id] == group_id as i32
                {
                    position_only_locks[vertex_id] = group_id as i32; // Then claim the vertex for this group
                } else {
                    position_only_locks[vertex_id] = -2; // Else vertex was already claimed by another group or was already locked, lock it
                }
            }
        }
    }

    // Lock vertices used by more than 1 group
    for i in 0..vertex_locks.len() {
        let vertex_id = position_only_vertex_remap[i] as usize;
        vertex_locks[i] = position_only_locks[vertex_id] == -2;
    }
}

fn simplify_meshlet_group(
    group_meshlets: &[usize],
    meshlets: &XMetal2Meshlets,
    vertices: &VertexDataAdapter<'_>,
    vertex_normals: &[f32],
    vertex_stride: usize,
    vertex_locks: &[bool],
) -> Option<(Vec<u32>, f32)> {
    // Build a new index buffer into the mesh vertex data by combining all meshlet data in the group
    let mut group_indices = Vec::new();
    for meshlet_id in group_meshlets {
        let meshlet = meshlets.get(*meshlet_id);
        for meshlet_index in meshlet.triangles {
            group_indices.push(meshlet.vertices[*meshlet_index as usize]);
        }
    }

    // Simplify the group to ~50% triangle count
    let mut error = 0.0;
    let simplified_group_indices = simplify_with_attributes_and_locks(
        &group_indices,
        vertices,
        vertex_normals,
        &[0.5; 3],
        vertex_stride,
        vertex_locks,
        group_indices.len() / 2,
        f32::MAX,
        SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute,
        Some(&mut error),
    );

    // Check if we were able to simplify at least a little
    if (simplified_group_indices.len() as f32 / group_indices.len() as f32)
        > SIMPLIFICATION_FAILURE_PERCENTAGE
    {
        return None;
    }

    // Convert error to object-space and convert from diameter to radius
    error *= simplify_scale(vertices) * 0.5;

    Some((simplified_group_indices, error))
}

/// An error produced by [`MeshletMesh::from_mesh`].
#[derive(Error, Display, Debug)]
pub enum MeshToMeshletMeshConversionError {
    #[display("Mesh primitive topology is not TriangleList")]
    WrongMeshPrimitiveTopology,
    #[display("Mesh attributes are not {{POSITION, NORMAL, UV_0}}")]
    WrongMeshVertexAttributes,
    #[display("Mesh has no indices")]
    MeshMissingIndices,
}
