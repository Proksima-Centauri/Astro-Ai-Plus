import deep_sky_catalog


def test_messier_catalog_contains_expected_number_of_objects():
    assert len(deep_sky_catalog.MESSIER_CATALOG) == 110


def test_messier_catalog_contains_m31_with_coordinates():
    m31 = next(item for item in deep_sky_catalog.MESSIER_CATALOG if item["name"] == "M31")

    assert m31["display_name"] == "Andromeda Galaxy"
    assert abs(m31["ra"] - 10.6847) < 1e-4
    assert abs(m31["dec"] - 41.2690) < 1e-4
    assert m31["catalog"] == "Messier"
