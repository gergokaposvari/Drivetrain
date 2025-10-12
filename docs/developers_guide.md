# Developer Guide

This document explains how the project is structured, where to modify gameplay or presentation logic, and how to extend the simulation with new tracks or features.

## Project Layout

- **`main.py`** – Program entry point. Parses CLI options (`--track`, `--list-tracks`), initializes pygame, selects the track, and runs the main loop. Modify this file to change command-line flags, window setup, or the overall frame cadence.
- **`src/topdown/`** – Core modules ported from the original pybox2d top-down car example, split into logical components:
  - `simulation.py`: Builds the Box2D world, spawns the chassis, loads track fixtures, and advances physics.
  - `render.py`: Draws the track surfaces, car, tires, grid, and the speedometer HUD using pygame primitives.
  - `car.py`: Sets up the chassis body and its revolute joints to the tires.
  - `tire.py`: Implements per-tire traction, friction, and drive/brake force.
  - `contact.py`: Contact listener that toggles tire traction when rolling over different surfaces.
  - `input.py`: Keyboard mapping from pygame events to the simple `up/down/left/right` control strings.
  - `track.py`: Data classes (`Track`, `TrackSurface`) describing the floor boundary, surfaces, and spawn point.
  - `track_loader.py`: Validates/loads JSON track definitions and discovers tracks in the `tracks/` folder.
- `track_builder.py`: Generates road polygons from spline control points and widths.
- `spline.py`: Catmull–Rom helper used by the track builder.
- `timing.py`: Manages mini-sector timing, lap validation, and best times.
  The builder also stores `sector_lines`, which are the perpendicular crossings from the left edge to the right edge; these are what the timing manager checks when a car completes a sector.

## Runtime Flow

1. `main.py` parses arguments and optionally lists discovered tracks in the `tracks/` directory.
2. Selected track JSON is loaded through `track_loader.load_track`:
   - If the file defines `control_points`/`widths`, the spline pipeline in `track_builder` generates the road surface on the fly.
   - Otherwise, the legacy `surfaces` array is respected verbatim.
3. `Simulation` constructs the Box2D world, registers the `TopDownContactListener`, spawns the `Car`, and creates static sensor fixtures for each surface polygon. The car is positioned at the declared `spawn_point`.
4. The main loop polls pygame events, hands them to `InputHandler`, advances the simulation with `Simulation.step`, and renders via `Renderer.draw`.
5. `Renderer.draw` centers the camera on the car body, draws the grid/background/boundary, iterates each surface polygon, renders the chassis and tires, and overlays the speedometer (km/h from Box2D linear velocity).

## Tracks and Assets

### Spline-based tracks (recommended)

Provide at least four control points and a matching `widths` array:

```json
{
  "spawn_point": [-150.0, -70.0],
  "spawn_direction": [0.83, 0.55],
  "control_points": [[-150, -70], [-40, -120], [70, -90], ...],
  "widths": [60, 60, 50, ...],
  "samples_per_segment": 8,            // optional
  "road_friction": 1.2,                // optional
  "grass_friction": 0.45,              // optional
  "margin": 20.0                       // optional
}
```

At load time the builder:

1. Creates a Catmull–Rom spline (closed loop) through the control points.
2. Samples the spline and offsets each point left/right by half the interpolated width.
3. Groups the resulting edge strips into Box2D-friendly convex polygons (respecting the 16-vertex limit).
4. Wraps the track in a rectangular grass surface with the requested margin.

Spawn orientation follows the spline tangent at the first control point when the car spawns at the provided `spawn_point`; you can override it by supplying `spawn_direction` in the JSON.

### Legacy polygon tracks

You can still provide surfaces manually:

```json
{
  "spawn_point": [x, y],
  "boundary": [[...], ...],
  "surfaces": [
    {
      "name": "road",
      "polygons": [[[...], ...]],
      "friction": 1.2,
      "fill_color": [r, g, b],
      "outline_color": [r, g, b]
    }
  ]
}
```

Each polygon must contain 3–16 vertices (Box2D limit). Stick with small monotone shapes; the loader does not attempt to triangulate concave surfaces.

### Bundled layouts

- `default.json` – Legacy hand-authored polygons.
- `classic_loop.json` – Spline-based loop demonstrating the new format.
- `hungaroring.json` – Polygonal approximation of the Hungaroring.

Preview PNGs (`*_preview.png`) sit next to their JSON counterparts for a quick sanity check. You can also launch the interactive editor with `uv run python -m src.editor.track_editor --output tracks/your_track.json` to place control points visually; left click adds points (default width 20 m), right click/`Z` removes, and `S` saves when four or more points are defined.

Add new tracks by dropping a JSON into `tracks/`. List options with `uv run python main.py --list-tracks`, then launch with `--track <name>` or pass an absolute path.

## Physics Customization

- **Chassis geometry** (`Car.vertices`, `Car.tire_anchors`) sets wheelbase and tire placement; update if designing different vehicles.
- **Steering limits**: `lock_angle`, `turn_speed_per_sec` in `Car.update` control max steering angle and rotation speed.
- **Tire handling** (`tire.py`) offers key tuning knobs: `max_drive_force`, `max_forward_speed`, `max_lateral_impulse`, etc.
- **Traction areas**: Each surface creates a sensor fixture tagged with `GroundArea.name` and `GroundArea.friction_modifier`. Tires switch traction via `Tire.add_ground_area` / `remove_ground_area`.

## Rendering & UI

- `RenderConfig` centralizes colors, resolution, and pixels-per-meter. Update to zoom in/out or recolor the scene.
- `_draw_grid`, `_draw_surfaces`, `_draw_car`, `_draw_speedometer` build the frame; extend these helpers to add lap counters, cameras, or debug overlays.

## Input

- Default keys: `W/S` accelerate/brake, `A/D` steer. Update `InputMapping.default` to remap, or extend `InputHandler.process_event` for gamepads.

## Extending the Track System

- To vary width along the course, edit the `widths` array; values interpolate smoothly between control points.
- Increase `samples_per_segment` for tighter fidelity on sharp bends (at the cost of more polygons).
- To add additional surfaces (e.g., gravel traps), append manual entries to `surfaces`; they are merged with the spline-generated road and grass.

## Lap Timing

- Timing is enabled when a track provides `control_points`/`widths`.
- A lap starts once the player presses any control input; each sector is bounded by consecutive control points.
- Leaving the asphalt (no tires on the road surface) immediately marks the lap invalid.
- Mini-sector times must be recorded for every sector before a lap is considered valid.
- The bottom HUD shows the live lap clock and a sector progress bar (green for faster than previous best, yellow for slower, red when invalid). Best lap/sector times are tracked per session.

## Build & Dependencies

- Dependencies are managed via `pyproject.toml`; run `uv sync` after editing.
- Core libraries: `pygame-ce` (render/input), `box2d` + `box2d-py` (physics), `matplotlib` (track previews).
- Run-time commands should use `uv run`, e.g., `uv run python main.py --track classic_loop`.

## Upgrading & Maintenance Tips

- Re-test physics after updating pygame or Box2D—changes to fixture handling may require tuning `tire.py` or `contact.py`.
- Keep modules focused; add new systems as separate files (e.g., `timing.py`) and plug them into `Simulation.step`.
- For additional HUD widgets, extend `Renderer` helpers rather than scattering rendering logic elsewhere.
- `Simulation.step` already receives the control set and timestep, making it the natural home for AI, lap counters, or replay code.

## Testing & Validation

- Quick validation: `uv run python main.py --track <track>` to visually inspect; `python -m compileall src` for syntax checks.
- Consider unit tests around `track_loader` (schema validation) and `Simulation` (fixture counts) using `pytest`.

## FAQ

1. **The car spawns off-track** – Adjust `spawn_point` in the JSON.
2. **Box2D complains about vertex counts** – Supply fewer control points or lower `samples_per_segment`; the builder chunks the strip but still respects the 16-vertex cap.
3. **Adding gravel or kerbs** – Append additional surfaces in the JSON with their own friction and colors.
4. **Switching controls to arrow keys** – Update the `InputMapping.default` dictionary in `input.py`.

Happy hacking!
