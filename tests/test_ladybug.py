from kvrrj.ladybug.location import Location, average_location, location_to_string


def test_location_to_string():
    loc = Location(
        city="Portobello",
        country="Naboombu",
        state="Liquid",
        latitude=51.5074,  # type: ignore
        longitude=0.1278,  # type: ignore
        elevation=-50,
        station_id="123",
        source="metoffice",
        time_zone=1,
    )
    assert location_to_string(loc) == f"{loc.country} - {loc.city}"


def test_average_location():
    vals = [0, 25, 50, 75]
    locs = [
        Location(latitude=i, longitude=i, elevation=i, city="A", country="B")
        for i in vals
    ]

    avg_loc = average_location(locs)

    assert avg_loc.latitude == sum(vals) / len(vals)
    assert avg_loc.longitude == sum(vals) / len(vals)
    assert avg_loc.elevation == sum(vals) / len(vals)
    assert avg_loc.city == "A|A|A|A"
    assert avg_loc.country == "B|B|B|B"
