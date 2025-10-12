"""Entry point running the refactored top-down car example with pygame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame

from src.topdown import TrackLoadError, discover_tracks, load_track
from src.topdown.input import InputHandler
from src.topdown.render import RenderConfig, Renderer
from src.topdown.simulation import Simulation

TIME_STEP = 1.0 / 60.0
VELOCITY_ITERATIONS = 6
POSITION_ITERATIONS = 2


def main() -> None:
    args = _parse_args()
    tracks_dir = Path(__file__).resolve().parent / "tracks"
    available_tracks = discover_tracks(tracks_dir)

    if args.list_tracks:
        _print_track_list(available_tracks)
        return

    selected_track = _resolve_track(args.track, available_tracks, tracks_dir)
    pygame.init()
    config = RenderConfig()
    screen = pygame.display.set_mode(config.screen_size)
    pygame.display.set_caption("Top Down Car (pybox2d example)")

    input_handler = InputHandler()
    clock = pygame.time.Clock()
    simulation = Simulation(track=selected_track)
    renderer = Renderer(screen, config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            else:
                input_handler.process_event(event)

        simulation.step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS, input_handler.active)
        renderer.draw(simulation)
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down car demo")
    parser.add_argument(
        "--track",
        help="Track name (from tracks directory) or path to a JSON file",
    )
    parser.add_argument(
        "--list-tracks",
        action="store_true",
        help="List bundled tracks and exit",
    )
    return parser.parse_args()


def _print_track_list(available_tracks) -> None:
    if not available_tracks:
        print("No tracks found.")
        return
    print("Available tracks:")
    for name, loaded in available_tracks.items():
        print(f"  {name:15s} -> {loaded.path}")


def _resolve_track(track_arg, available_tracks, tracks_dir: Path):
    if track_arg is None:
        return None

    candidate_path = Path(track_arg)
    if candidate_path.exists():
        try:
            return load_track(candidate_path)
        except TrackLoadError as exc:
            print(f"Warning: {exc}; falling back to default track.")
            return None

    if track_arg in available_tracks:
        return available_tracks[track_arg].track

    candidate_file = tracks_dir / f"{track_arg}.json"
    if candidate_file.exists():
        try:
            return load_track(candidate_file)
        except TrackLoadError as exc:
            print(f"Warning: {exc}; falling back to default track.")
            return None

    print(f"Warning: track '{track_arg}' not found; using default track.")
    return None


if __name__ == "__main__":
    main()
