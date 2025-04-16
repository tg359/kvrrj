from ladybug_geometry.geometry2d import Mesh2D, Point2D
from ladybug_geometry.geometry3d import Mesh3D, Plane, Point3D


def points_to_mesh3d(points: list[Point3D], alpha: float) -> Mesh3D:
    """Convert a list of points to a ladybug-geometry Mesh3D.

    Args:
        points (list[Point3D]):
            A list of ladybug-geometry Point3D objects.
        alpha (float):
            The alpha value for the mesh. Mesh faces with edges not within this
            tolerance will be removed.

    Returns:
        Mesh3D: A ladybug-geometry Mesh3D.
    """
    raise NotImplementedError()
    return Mesh3D()


def project_mesh3d_to_mesh2d(mesh: Mesh3D, plane: Plane = Plane()) -> Mesh2D:
    """Project a mesh to a 2D plane.

    Args:
        mesh (Mesh3D):
            A ladybug-geometry Mesh3D object.
        plane (Plane):
            The plane to project the mesh to.
            Defaults to the XY plane.

    Returns:
        Mesh2D: A ladybug-geometry Mesh2D object.
    """

    projected_vertices = [plane.project_point(pt) for pt in mesh.vertices]
    projected_vertices_2d = [Point2D(*i.to_array()[:-1]) for i in projected_vertices]
    return Mesh2D(vertices=projected_vertices_2d, faces=mesh.faces, colors=mesh.colors)
