from math import acos, cos, sin

from ladybug_geometry.geometry3d import Mesh3D, Point3D, Vector3D


def create_icosphere(resolution: int = 1) -> Mesh3D:
    """Create an icosphere mesh with the given resolution.

    Args:
        resolution (int, optional):
            The number of subdivisions for the icosphere.
            Default is 1.

    Returns:
        Mesh3D:
            The resulting icosphere mesh.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")
    if not isinstance(resolution, int):
        raise ValueError("Resolution must be an integer.")

    def slerp(start: Vector3D, end: Vector3D, t: float) -> Vector3D:
        """Spherical linear interpolation."""
        dot = max(-1.0, min(1.0, start.dot(end)))  # Clamp dot product to avoid errors
        theta = acos(dot) * t
        relative_vec = end - start * dot
        relative_vec = relative_vec.normalize()
        return start * cos(theta) + relative_vec * sin(theta)

    # Base vertices of an icosahedron
    phi = (1 + 5**0.5) / 2  # Golden ratio
    base_vertices = [
        Vector3D(-1, phi, 0).normalize(),
        Vector3D(1, phi, 0).normalize(),
        Vector3D(-1, -phi, 0).normalize(),
        Vector3D(1, -phi, 0).normalize(),
        Vector3D(0, -1, phi).normalize(),
        Vector3D(0, 1, phi).normalize(),
        Vector3D(0, -1, -phi).normalize(),
        Vector3D(0, 1, -phi).normalize(),
        Vector3D(phi, 0, -1).normalize(),
        Vector3D(phi, 0, 1).normalize(),
        Vector3D(-phi, 0, -1).normalize(),
        Vector3D(-phi, 0, 1).normalize(),
    ]

    # Base faces of an icosahedron (triangles)
    base_faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    # Initialize vertices and faces
    vertices = base_vertices[:]
    faces = base_faces[:]

    # Subdivide each face
    for _ in range(resolution):
        new_faces = []
        midpoints = {}

        def get_midpoint(v1, v2):
            """Get or create the midpoint of two vertices."""
            smaller, larger = min(v1, v2), max(v1, v2)
            key = (smaller, larger)
            if key not in midpoints:
                midpoint = slerp(vertices[smaller], vertices[larger], 0.5).normalize()
                midpoints[key] = len(vertices)
                vertices.append(midpoint)
            return midpoints[key]

        for v1, v2, v3 in faces:
            # Split each edge of the triangle
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)

            # Create four new faces
            new_faces.append((v1, a, c))
            new_faces.append((a, v2, b))
            new_faces.append((c, b, v3))
            new_faces.append((a, b, c))

        faces = new_faces

    # Convert vertices to Point3D and create the mesh
    points = [Point3D(v.x, v.y, v.z) for v in vertices]
    return Mesh3D(vertices=points, faces=faces)
