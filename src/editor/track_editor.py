"""Interactive spline-based track editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import pygame

from src.topdown.track_builder import SplineTrackConfig, build_track_from_spline
from src.topdown.track import Track

Vec2 = Tuple[float, float]

WINDOW_SIZE = (1280, 720)
BACKGROUND_COLOR = (24, 30, 32)
GRID_COLOR = (60, 68, 72)
CONTROL_COLOR = (235, 110, 40)
CONTROL_LINE_COLOR = (200, 200, 210)
TEXT_COLOR = (240, 240, 240)
ROAD_ALPHA = 220

DEFAULT_WIDTH = 30.0
DEFAULT_MARGIN = 25.0
DEFAULT_SAMPLES_PER_SEGMENT = 10
PIXELS_PER_METER = 5.0
CAMERA_STEP = 40.0  # pixels per keypress


def world_to_screen(point: Vec2, center: Vec2) -> Tuple[int, int]:
    screen_x = int(center[0] + point[0] * PIXELS_PER_METER)
    screen_y = int(center[1] - point[1] * PIXELS_PER_METER)
    return screen_x, screen_y


def screen_to_world(point: Tuple[int, int], center: Vec2) -> Vec2:
    world_x = (point[0] - center[0]) / PIXELS_PER_METER
    world_y = (center[1] - point[1]) / PIXELS_PER_METER
    return world_x, world_y


def render_background(surface: pygame.Surface, center: Vec2) -> None:
    surface.fill(BACKGROUND_COLOR)
    width, height = surface.get_size()
    num_lines_x = int(width / (PIXELS_PER_METER * 10)) + 2
    num_lines_y = int(height / (PIXELS_PER_METER * 10)) + 2
    offset_x = int(center[0] % (PIXELS_PER_METER * 10))
    offset_y = int(center[1] % (PIXELS_PER_METER * 10))
    for i in range(-num_lines_x, num_lines_x + 1):
        x = offset_x + i * PIXELS_PER_METER * 10
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, height), 1)
    for j in range(-num_lines_y, num_lines_y + 1):
        y = offset_y + j * PIXELS_PER_METER * 10
        pygame.draw.line(surface, GRID_COLOR, (0, y), (width, y), 1)


def render_track(surface: pygame.Surface, track: Track, center: Vec2) -> None:
    if track is None:
        return
    road_surface = next((s for s in track.surfaces if s.name == "road"), None)
    grass_surface = next((s for s in track.surfaces if s.name == "grass"), None)

    if grass_surface:
        for polygon in grass_surface.polygons:
            points = [world_to_screen(point, center) for point in polygon]
            pygame.draw.polygon(surface, grass_surface.fill_color, points)
            pygame.draw.polygon(surface, grass_surface.outline_color, points, width=2)

    if road_surface:
        road_color = road_surface.fill_color + (ROAD_ALPHA,)
        road_outline = road_surface.outline_color
        temp = pygame.Surface(surface.get_size(), flags=pygame.SRCALPHA)
        for polygon in road_surface.polygons:
            points = [world_to_screen(point, center) for point in polygon]
            pygame.draw.polygon(temp, road_color, points)

            if road_outline is not None:
                pygame.draw.polygon(surface, road_outline, points, width=2)
        surface.blit(temp, (0, 0))


def render_control_points(
    surface: pygame.Surface,
    points: Sequence[Vec2],
    font: pygame.font.Font,
    center: Vec2,
) -> None:
    if not points:
        return
    screen_points = [world_to_screen(point, center) for point in points]
    if len(screen_points) >= 2:
        pygame.draw.lines(surface, CONTROL_LINE_COLOR, True, screen_points, 2)
    for idx, (world_point, screen_point) in enumerate(zip(points, screen_points)):
        pygame.draw.circle(surface, CONTROL_COLOR, screen_point, 6)
        label = font.render(str(idx), True, TEXT_COLOR)
        surface.blit(label, (screen_point[0] + 8, screen_point[1] - 6))


def render_overlay(
    surface: pygame.Surface, font: pygame.font.Font, messages: Sequence[str]
) -> None:
    y = 10
    for message in messages:
        text = font.render(message, True, TEXT_COLOR)
        surface.blit(text, (10, y))
        y += text.get_height() + 4


def save_track(
    path: Path, control_points: Sequence[Vec2], widths: Sequence[float]
) -> None:
    spawn_dir = [0.0, 1.0]
    if len(control_points) >= 2:
        dx = control_points[1][0] - control_points[0][0]
        dy = control_points[1][1] - control_points[0][1]
        length = (dx * dx + dy * dy) ** 0.5
        if length != 0:
            spawn_dir = [dx / length, dy / length]
    data = {
        "spawn_point": [control_points[0][0], control_points[0][1]],
        "spawn_direction": spawn_dir,
        "control_points": [[x, y] for x, y in control_points],
        "widths": list(widths),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def build_preview_track(
    control_points: Sequence[Vec2], widths: Sequence[float]
) -> Track | None:
    if len(control_points) < 4:
        return None
    config = SplineTrackConfig(
        control_points=control_points,
        widths=widths,
        spawn_point=control_points[0],
        samples_per_segment=DEFAULT_SAMPLES_PER_SEGMENT,
        margin=DEFAULT_MARGIN,
    )
    return build_track_from_spline(config)


def run_editor(output_path: Path | None) -> None:
    pygame.init()
    pygame.display.set_caption("Track Editor")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    center = [WINDOW_SIZE[0] / 2.0, WINDOW_SIZE[1] / 2.0]
    control_points: List[Vec2] = []
    widths: List[float] = []
    is_panning = False
    pan_origin = (0, 0)
    center_origin = (center[0], center[1])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif (
                    event.key == pygame.K_s and len(control_points) >= 4 and output_path
                ):
                    save_track(output_path, control_points, widths)
                elif event.key == pygame.K_z and control_points:
                    control_points.pop()
                    widths.pop()
                elif event.key == pygame.K_LEFT:
                    center[0] += CAMERA_STEP
                elif event.key == pygame.K_RIGHT:
                    center[0] -= CAMERA_STEP
                elif event.key == pygame.K_UP:
                    center[1] += CAMERA_STEP
                elif event.key == pygame.K_DOWN:
                    center[1] -= CAMERA_STEP
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    world_point = screen_to_world(event.pos, center)
                    control_points.append(world_point)
                    widths.append(DEFAULT_WIDTH)
                elif event.button == 3 and control_points:
                    control_points.pop()
                    widths.pop()
                elif event.button == 2:
                    is_panning = True
                    pan_origin = event.pos
                    center_origin = (center[0], center[1])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                is_panning = False
            elif event.type == pygame.MOUSEMOTION and is_panning:
                dx = event.pos[0] - pan_origin[0]
                dy = event.pos[1] - pan_origin[1]
                center[0] = center_origin[0] + dx
                center[1] = center_origin[1] + dy

        render_background(screen, center)
        track = build_preview_track(control_points, widths)
        if track:
            render_track(screen, track, center)
        render_control_points(screen, control_points, font, center)
        overlay = [
            "Left click: add control point",
            "Right click / Z: remove last point",
            "Middle click or arrow keys: pan",
            "S: save track" if output_path else "",
            "ESC: exit",
            f"Control points: {len(control_points)}",
        ]
        overlay = [line for line in overlay if line]
        render_overlay(screen, font, overlay)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive spline-based track editor"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file. Press 'S' in the editor to save.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive tool
    args = parse_args()
    run_editor(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
