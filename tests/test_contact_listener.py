from types import SimpleNamespace

from Box2D import b2World

from src.topdown.car import Car
from src.topdown.contact import TopDownContactListener
from src.topdown.ground import GroundArea


def _make_contact(obj_a, obj_b):
    body_a = SimpleNamespace(userData={"obj": obj_a})
    body_b = SimpleNamespace(userData={"obj": obj_b})
    fixture_a = SimpleNamespace(body=body_a)
    fixture_b = SimpleNamespace(body=body_b)
    return SimpleNamespace(fixtureA=fixture_a, fixtureB=fixture_b)


def test_tire_stays_on_surface_while_any_fixture_overlaps():
    world = b2World(gravity=(0.0, 0.0))
    car = Car(world)
    tire = car.tires[0]
    listener = TopDownContactListener()
    road = GroundArea(name="road", friction_modifier=1.2)

    contact = _make_contact(tire, road)

    # Two overlapping fixtures should still count as a single surface contact.
    listener._handle_contact(contact, True)
    listener._handle_contact(contact, True)
    assert any(area.name == "road" for area in tire.ground_areas)

    # Dropping one fixture must not remove the surface.
    listener._handle_contact(contact, False)
    assert any(area.name == "road" for area in tire.ground_areas)

    # Once the last overlapping fixture ends the surface disappears.
    listener._handle_contact(contact, False)
    assert all(area.name != "road" for area in tire.ground_areas)
