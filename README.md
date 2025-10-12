# Top-Down Car Sandbox

This project wraps the pybox2d top-down car example into a small pygame sandbox. You can choose from bundled circuits or describe new ones with a smooth centerline spline.

## Running the demo

```bash
uv run python main.py --list-tracks     # show bundled layouts
uv run python main.py --track default   # run the default course
uv run python main.py --track classic_loop
```

Any `.json` file in the `tracks/` directory is discovered at startup. You can also pass an absolute path with `--track /path/to/track.json`.

## Interactive Track Editor

Sketch a new circuit by placing spline control points:

```bash
uv run python -m src.editor.track_editor --output tracks/my_new_track.json
```

- Left click adds a control point (width defaults to 20 m).
- Right click (or `Z`) removes the last point.
- Press `S` to save once you have at least four points.
- The preview updates in real time—tweak widths later directly in the JSON file if needed.

## Authoring tracks with splines

Define the track centreline with a sequence of control points and matching widths:

```json
{
  "spawn_point": [-150.0, -70.0],
  "spawn_direction": [0.83, 0.55],
  "control_points": [
    [-150.0, -70.0],
    [-40.0, -120.0],
    [70.0, -90.0],
    [130.0, 0.0],
    [60.0, 90.0],
    [-40.0, 100.0],
    [-140.0, 40.0],
    [-180.0, -40.0]
  ],
  "widths": [
    60.0,
    60.0,
    50.0,
    55.0,
    60.0,
    55.0,
    60.0,
    60.0
  ]
}
```

At load time the engine:

1. Builds a Catmull–Rom spline through the control points (closed loop).
2. Offsets the centreline left/right by half the width to create road edges.
3. Generates the asphalt surface using the fewest Box2D-friendly polygons it can, and wraps everything in a grass rectangle.

The number of control points and the `widths` array must match, and you need at least four control points for a closed track. Width values are in metres.

Optional keys:

- `samples_per_segment` (default `8`) – spline sampling density.
- `road_friction` / `grass_friction` – traction modifiers.
- `margin` – additional grass border around the generated road.

Legacy polygon-based tracks still load; any file that defines a `surfaces` array bypasses the spline pipeline.

## Previewing tracks

Preview PNGs for bundled tracks live alongside their JSON definitions (for example `tracks/classic_loop_preview.png`). They are generated with matplotlib so you can visually verify the shape before loading it in-game.

## Lap Timing

When a spline track is loaded, the game divides the circuit into mini sectors—one between each pair of control points. Timing starts once you press any input key and resets automatically after every lap. Magenta debug lines span the track from edge to edge for each sector; crossing them records the split. The HUD at the bottom shows the running lap time, best/last laps, and a sector bar that lights up green when you beat your previous best or yellow when you are slower.
