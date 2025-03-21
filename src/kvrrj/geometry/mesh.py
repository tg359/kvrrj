from ladybug_geometry.geometry3d import Mesh3D, Point3D


def points_to_mesh(points: list[Point3D], alpha: float) -> Mesh3D:
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
    return Mesh3D()
