"""Pygame rendering for the refactored top-down car sample."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import pygame
from Box2D import b2Vec2  # type: ignore import-untyped

from .simulation import Simulation
from .timing import TimingState


@dataclass(frozen=True)
class RenderConfig:
    screen_size: Tuple[int, int] = (1280, 720)
    pixels_per_meter: float = 5
    background_color: Tuple[int, int, int] = (25, 25, 30)
    car_color: Tuple[int, int, int] = (220, 80, 60)
    car_outline_color: Tuple[int, int, int] = (250, 240, 230)
    tire_color: Tuple[int, int, int] = (30, 30, 30)
    tire_outline_color: Tuple[int, int, int] = (200, 200, 200)
    grid_color: Tuple[int, int, int] = (50, 50, 60)
    grid_spacing: float = 5.0
    boundary_color: Tuple[int, int, int] = (180, 180, 200)
    ground_area_color: Tuple[int, int, int] = (60, 100, 60)
    ground_area_outline: Tuple[int, int, int] = (80, 150, 80)
    hud_text_color: Tuple[int, int, int] = (245, 245, 245)
    hud_background: Tuple[int, int, int] = (15, 15, 18)
    timing_invalid_color: Tuple[int, int, int] = (200, 70, 70)
    sector_pending_color: Tuple[int, int, int] = (70, 70, 80)
    sector_fast_color: Tuple[int, int, int] = (60, 180, 120)
    sector_slow_color: Tuple[int, int, int] = (220, 200, 40)
    sector_background_color: Tuple[int, int, int] = (35, 35, 40)
    sector_line_color: Tuple[int, int, int] = (200, 80, 200)
    draw_sensor_rays: bool = False
    sensor_ray_hit_color: Tuple[int, int, int] = (240, 200, 40)
    sensor_ray_miss_color: Tuple[int, int, int] = (150, 80, 80)
    sensor_ray_width: int = 2


class Renderer:
    """Draws the car and environment."""

    def __init__(
        self, screen: pygame.Surface, config: RenderConfig | None = None
    ) -> None:
        self.screen = screen
        self.config = config or RenderConfig()
        self._half_width = self.config.screen_size[0] / 2.0
        self._half_height = self.config.screen_size[1] / 2.0
        self._font = pygame.font.SysFont("Arial", 16)
        self._hud_font = pygame.font.SysFont("Arial", 24)
        self._hud_small_font = pygame.font.SysFont("Arial", 14)
        display_surface = pygame.display.get_surface() if pygame.display.get_init() else None
        self._flip_display = display_surface is not None and display_surface == screen

    def draw(self, simulation: Simulation) -> None:
        car = simulation.car
        focus = car.body.worldCenter
        self.screen.fill(self.config.background_color)
        self._draw_grid(focus)
        self._draw_surfaces(simulation.surfaces, focus)
        self._draw_timing_segments(simulation.timing_segments, focus)
        self._draw_boundaries(simulation.boundary_segments, focus)
        self._draw_car(car, focus)
        if self.config.draw_sensor_rays:
            self._draw_sensor_rays(simulation, focus)
        self._draw_speedometer(car)
        self._draw_timing(simulation.timing_state)
        if self._flip_display:
            pygame.display.flip()

    def _draw_grid(self, focus: b2Vec2) -> None:
        spacing = self.config.grid_spacing
        if spacing <= 0:
            return
        ppm = self.config.pixels_per_meter
        min_x = focus.x - self._half_width / ppm
        max_x = focus.x + self._half_width / ppm
        min_y = focus.y - self._half_height / ppm
        max_y = focus.y + self._half_height / ppm
        start_col = int(min_x // spacing) - 1
        end_col = int(max_x // spacing) + 1
        start_row = int(min_y // spacing) - 1
        end_row = int(max_y // spacing) + 1
        for column in range(start_col, end_col + 1):
            world_x = column * spacing
            start = self._world_to_screen(b2Vec2(world_x, min_y), focus)
            end = self._world_to_screen(b2Vec2(world_x, max_y), focus)
            pygame.draw.line(self.screen, self.config.grid_color, start, end, 1)
        for row in range(start_row, end_row + 1):
            world_y = row * spacing
            start = self._world_to_screen(b2Vec2(min_x, world_y), focus)
            end = self._world_to_screen(b2Vec2(max_x, world_y), focus)
            pygame.draw.line(self.screen, self.config.grid_color, start, end, 1)

    def _draw_car(self, car, focus: b2Vec2) -> None:
        self._draw_polygon_body(
            car.body,
            car.vertices,
            self.config.car_color,
            self.config.car_outline_color,
            focus,
        )
        for tire in car.tires:
            fixture = next(iter(tire.body.fixtures))
            shape = fixture.shape
            verts = list(shape.vertices)
            self._draw_polygon_body(
                tire.body,
                verts,
                self.config.tire_color,
                self.config.tire_outline_color,
                focus,
            )

    def _draw_polygon_body(
        self,
        body,
        local_vertices: Sequence[Tuple[float, float] | b2Vec2],
        fill_color: Tuple[int, int, int],
        outline_color: Tuple[int, int, int],
        focus: b2Vec2,
    ) -> None:
        vertices = [body.GetWorldPoint(vertex) for vertex in local_vertices]
        points = [self._world_to_screen(vertex, focus) for vertex in vertices]
        pygame.draw.polygon(self.screen, fill_color, points)
        pygame.draw.polygon(self.screen, outline_color, points, 2)

    def _world_to_screen(self, point: b2Vec2, focus: b2Vec2) -> Tuple[int, int]:
        ppm = self.config.pixels_per_meter
        dx = (point.x - focus.x) * ppm
        dy = (point.y - focus.y) * ppm
        screen_x = int(self._half_width + dx)
        screen_y = int(self._half_height - dy)
        return screen_x, screen_y

    def _draw_boundaries(self, segments, focus: b2Vec2) -> None:
        color = self.config.boundary_color
        for start, end in segments:
            start_pt = self._world_to_screen(start, focus)
            end_pt = self._world_to_screen(end, focus)
            pygame.draw.line(self.screen, color, start_pt, end_pt, 2)

    def _draw_surfaces(self, surfaces, focus: b2Vec2) -> None:
        for surface, polygons in surfaces:
            for polygon in polygons:
                points = [self._world_to_screen(vertex, focus) for vertex in polygon]
                pygame.draw.polygon(self.screen, surface.fill_color, points)
                if surface.outline_color is not None:
                    pygame.draw.polygon(self.screen, surface.outline_color, points, 2)

    def _draw_timing_segments(self, segments, focus: b2Vec2) -> None:
        if not segments:
            return
        color = self.config.sector_line_color
        for start, end in segments:
            start_pt = self._world_to_screen(b2Vec2(*start), focus)
            end_pt = self._world_to_screen(b2Vec2(*end), focus)
            pygame.draw.line(self.screen, color, start_pt, end_pt, 1)

    def _draw_sensor_rays(self, simulation: Simulation, focus: b2Vec2) -> None:
        sample = simulation.sample_sensors()
        if not sample.distances:
            return

        car = simulation.car
        origin = car.body.worldCenter
        forward = car.forward_vector
        right = b2Vec2(-forward.y, forward.x)

        angles = simulation.sensor_angles()
        max_distance = simulation.sensor_max_distance()

        for angle_deg, distance, hit in zip(angles, sample.distances, sample.hits):
            theta = math.radians(angle_deg)
            direction = forward * math.cos(theta) + right * math.sin(theta)
            if direction.length == 0:
                continue
            direction.Normalize()

            length = min(distance, max_distance)
            end_point = origin + direction * length

            start_screen = self._world_to_screen(origin, focus)
            end_screen = self._world_to_screen(end_point, focus)
            color = (
                self.config.sensor_ray_hit_color if hit else self.config.sensor_ray_miss_color
            )
            pygame.draw.line(
                self.screen,
                color,
                start_screen,
                end_screen,
                self.config.sensor_ray_width,
            )

    def _draw_speedometer(self, car) -> None:
        velocity = car.tires[0].forward_velocity
        speed_mps = velocity.length
        speed_kmh = speed_mps
        label = f"Speed: {speed_kmh:5.1f} km/h"
        text_surface = self._hud_small_font.render(
            label, True, self.config.hud_text_color
        )
        padding = 8
        rect = text_surface.get_rect()
        rect.topleft = (padding, padding)
        background_rect = rect.inflate(padding * 2, padding)
        background_rect.topleft = (padding // 2, padding // 2)
        pygame.draw.rect(self.screen, self.config.hud_background, background_rect)
        self.screen.blit(text_surface, rect)

    def _draw_timing(self, state: TimingState | None) -> None:
        if state is None:
            return

        width, height = self.config.screen_size
        margin = 16

        time_color = (
            self.config.hud_text_color
            if not state.invalid
            else self.config.timing_invalid_color
        )
        time_text = self._hud_font.render(
            _format_time(state.current_time), True, time_color
        )
        time_rect = time_text.get_rect()
        time_rect.midbottom = (width // 2, height - margin - 42)
        self.screen.blit(time_text, time_rect)

        if state.best_lap_time is not None:
            best_text = self._hud_small_font.render(
                f"Best: {_format_time(state.best_lap_time)}",
                True,
                self.config.hud_text_color,
            )
            best_rect = best_text.get_rect()
            best_rect.topright = (width - margin, margin)
            self.screen.blit(best_text, best_rect)

        if state.last_lap_time is not None:
            last_text = self._hud_small_font.render(
                f"Last: {_format_time(state.last_lap_time)}",
                True,
                self.config.hud_text_color,
            )
            last_rect = last_text.get_rect()
            last_rect.topleft = (margin * 10, margin)
            self.screen.blit(last_text, last_rect)

        sectors = state.sector_statuses
        if not sectors:
            return

        bar_width = width - margin * 2
        bar_height = 16
        bar_rect = pygame.Rect(
            margin, height - margin - bar_height, bar_width, bar_height
        )
        pygame.draw.rect(self.screen, self.config.sector_background_color, bar_rect)

        segment_width = bar_width / len(sectors)
        for index, sector in enumerate(sectors):
            segment_rect = pygame.Rect(
                margin + int(index * segment_width),
                height - margin - bar_height,
                int(segment_width) + 1,
                bar_height,
            )
            if not sector.completed:
                color = self.config.sector_pending_color
            else:
                color = (
                    self.config.sector_fast_color
                    if sector.faster
                    else self.config.sector_slow_color
                )
                if state.invalid:
                    color = self.config.timing_invalid_color
            pygame.draw.rect(self.screen, color, segment_rect)
            pygame.draw.rect(
                self.screen, self.config.background_color, segment_rect, width=1
            )


__all__ = ["Renderer", "RenderConfig"]


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.--"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes:02d}:{remainder:06.3f}"
