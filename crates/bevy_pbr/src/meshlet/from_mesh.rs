use super::asset::MeshletMesh;
use bevy_render::mesh::Mesh;

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
        _mesh: &Mesh,
        _denorm_scale: [f32; 3],
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        todo!()
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
