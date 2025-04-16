import math

from honeybee.model import (
    Aperture,
    Door,
    Face,
    Model,
    RoofCeiling,
    Room,
    Shade,
    ShadeMesh,
    Wall,
)
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry2d import (
    Arc2D,
    LineSegment2D,
    Mesh2D,
    Point2D,
    Polygon2D,
    Polyline2D,
    Ray2D,
    Vector2D,
)
from ladybug_geometry.geometry3d import (
    Arc3D,
    Face3D,
    LineSegment3D,
    Mesh3D,
    Plane,
    Point3D,
    Polyface3D,
    Polyline3D,
    Ray3D,
    Vector3D,
)

POINT2D = Point2D(1, 2)
POINT3D = Point3D(1, 2, 3)
VECTOR2D = Vector2D(3, 4)
VECTOR3D = Vector3D(3, 4, 5)
ARC2D_OPEN = Arc2D(Point2D(0, 0), 5, 0, 1 * math.pi)
ARC2D_CLOSED = Arc2D(Point2D(0, 0), 5, 0, 2 * math.pi)
ARC3D_OPEN = Arc3D(Plane(), 5, 0, 1 * math.pi)
ARC3D_CLOSED = Arc3D(Plane(), 5, 0, 2 * math.pi)
RAY2D = Ray2D(Point2D(0, 0), Vector2D(1, 1))
RAY3D = Ray3D(Point3D(0, 0, 0), Vector3D(1, 1, 1))
SEGMENT2D = LineSegment2D(Point2D(0, 0), Point2D(3, 4))
SEGMENT3D = LineSegment3D(Point3D(0, 0, 0), Point3D(4, 5, 6))
POLYLINE2D = Polyline2D([Point2D(0, 0), Point2D(3, 4), Point2D(5, 6)])
POLYLINE3D = Polyline3D([Point3D(0, 0, 0), Point3D(4, 5, 6), Point3D(7, 8, 9)])
POLYGON2D = Polygon2D([Point2D(0, 0), Point2D(3, 4), Point2D(5, 6)])
FACE3D = Face3D.from_regular_polygon(side_count=5)
FACE3D_HOLEY = Face3D.from_punched_geometry(
    base_face=Face3D.from_regular_polygon(side_count=4, radius=5),
    sub_faces=[Face3D.from_regular_polygon(side_count=3, radius=2)],
)
POLYFACE3D = Polyface3D.from_faces(
    [Face3D.from_regular_polygon(side_count=5)], tolerance=0.1
)
MESH2D = Mesh2D.from_face_vertices(
    faces=[
        Polygon2D([Point2D(0, 0), Point2D(0, 1), Point2D(1, 1)]),
        Polygon2D([Point2D(0, 0), Point2D(1, 1), Point2D(1, 0)]),
    ]
)
MESH3D = Mesh3D.from_face_vertices(
    faces=[
        Face3D([Point3D(0, 0, 0), Point3D(0, 1, 0), Point3D(1, 1, 0)]),
        Face3D([Point3D(0, 0, 0), Point3D(1, 1, 0), Point3D(1, 0, 0)]),
    ]
)

SHADE = Shade(identifier="shade", geometry=FACE3D)
SHADEMESH = ShadeMesh(identifier="shademesh", geometry=MESH3D)
DOOR = Door(identifier="door", geometry=FACE3D)
APERTURE = Aperture(identifier="aperture", geometry=FACE3D)
WALL_INTERIOR = Face(identifier="wall", geometry=FACE3D)
WALL_EXTERIOR = Face(identifier="wall", geometry=FACE3D)
WALL_GROUND = Face(identifier="wall", geometry=FACE3D)
ROOF_EXTERIOR = Face(identifier="roof", geometry=FACE3D)
ROOF_INTERIOR = Face(identifier="ceiling", geometry=FACE3D)
ROOF_GROUND = Face(identifier="roof", geometry=FACE3D)
FLOOR_EXTERIOR = Face(identifier="floor", geometry=FACE3D)
FLOOR_INTERIOR = Face(identifier="floor", geometry=FACE3D)
FLOOR_GROUND = Face(identifier="floor", geometry=FACE3D)

SENSOR = Sensor.from_raw_values()
SENSORGRID = SensorGrid.from_face3d(
    identifier="sensorgrid", faces=[FACE3D], x_dim=0.25, flip=True, offset=0.6
)

# complex objects
ROOM = Room.from_box(identifier="room", width=4, depth=4, height=3)
for n, face in enumerate(ROOM.faces):
    face: Face
    # add apertures
    if isinstance(face.type, Wall):
        face.apertures_by_ratio(0.4)
    if isinstance(face.type, RoofCeiling):
        face.apertures_by_ratio(0.2)
    # convert one aperture to door
    if "Front" in face.identifier:
        door = Door(identifier="door", geometry=face.apertures[0].geometry)
        face.remove_sub_faces()
        face.add_door(door)
# add shades
outdoor_shade = Shade(
    identifier="outdoor_shade",
    geometry=Face3D.from_regular_polygon(
        side_count=5, base_plane=Plane(o=Point3D(5, 5, 3.5))
    ),
)
indoor_shade = Shade(
    identifier="indoor_shade",
    geometry=Face3D.from_regular_polygon(
        side_count=3, base_plane=Plane(o=Point3D(2, 2, 0.75))
    ),
)
ROOM.add_outdoor_shade(outdoor_shade)
ROOM.add_indoor_shade(indoor_shade)
SENSORGRID = SensorGrid.from_face3d(
    identifier="sensorgrid",
    faces=[ROOM.faces[0].geometry],
    x_dim=0.25,
    offset=0.75,
    flip=True,
)
MODEL = Model.from_objects(identifier="model", objects=[ROOM])
MODEL.properties.radiance.add_sensor_grids([SENSORGRID])
