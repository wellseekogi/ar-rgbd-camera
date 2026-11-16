import random
from dataclasses import dataclass


@dataclass
class GameConfig:
    rows: int = 4
    cols: int = 4
    cell_m: float = 0.45
    origin_x: float = -0.8
    origin_y: float = 0.35
    lives: int = 3
    duration_s: float = 60.0
    warning_s: float = 3.0
    min_warning_s: float = 1.0
    lava_s: float = 2.0
    gap_s: float = 1.0

    def validate(self):
        import math

        if type(self.rows) is not int or type(self.cols) is not int:
            raise ValueError("격자 행·열은 정수여야 해요.")
        if not 2 <= self.rows <= 8 or not 2 <= self.cols <= 8:
            raise ValueError("격자는 2~8행·열로 설정해 주세요.")
        values = (
            self.cell_m,
            self.origin_x,
            self.origin_y,
            self.duration_s,
            self.warning_s,
            self.min_warning_s,
            self.lava_s,
            self.gap_s,
        )
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("게임 설정은 유한한 숫자여야 해요.")
        if not 0.2 <= self.cell_m <= 1.5 or max(abs(self.origin_x), abs(self.origin_y)) > 10:
            raise ValueError("칸 크기 또는 시작 위치가 허용 범위를 벗어났어요.")
        if type(self.lives) is not int or not 1 <= self.lives <= 10:
            raise ValueError("생명은 1~10개로 설정해 주세요.")
        if not 10 <= self.duration_s <= 600:
            raise ValueError("제한 시간은 10~600초로 설정해 주세요.")
        if not 0.5 <= self.min_warning_s <= self.warning_s <= 10:
            raise ValueError("경고 시간 범위를 확인해 주세요.")
        if not 0.5 <= self.lava_s <= 10 or not 0.2 <= self.gap_s <= 10:
            raise ValueError("용암·휴식 시간을 확인해 주세요.")


class Game:
    def __init__(self, config=None, rng=None):
        self.config = config or GameConfig()
        self.config.validate()
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.status, self.phase = "ready", "gap"
        self.remaining = self.config.duration_s
        self.phase_left = self.config.gap_s
        self.lives = self.config.lives
        self.cells = []
        self.round = 0
        self.last_hit = -10.0
        self.elapsed = 0.0
        self.reason = ""

    def start(self):
        self.reset()
        self.status = "playing"

    def cell(self, foot):
        cfg = self.config
        x = (foot[0] - cfg.origin_x) / cfg.cell_m
        y = (foot[1] - cfg.origin_y) / cfg.cell_m
        if not 0 <= x < cfg.cols or not 0 <= y < cfg.rows:
            return None
        return int(y) * cfg.cols + int(x)

    def step(self, dt, tracking):
        if self.status not in ("playing", "paused"):
            return
        feet = tracking.get("feet", [])
        occupied = [self.cell(foot) for foot in feet]
        if not tracking.get("valid") or len(feet) != 2 or None in occupied:
            self.status = "paused"
            self.reason = tracking.get("reason") or "양발이 게임판 안에 있어야 해요."
            return
        self.status, self.reason = "playing", ""
        # No accumulated damage after a suspended event loop or missing camera feed.
        dt = max(0.0, min(dt, 0.1))
        self.remaining = max(0.0, self.remaining - dt)
        self.elapsed += dt
        self.phase_left -= dt
        if self.remaining <= 0:
            self.status = "won"
            return
        if self.phase_left <= 0:
            cfg = self.config
            if self.phase == "gap":
                self.phase = "warning"
                self.round += 1
                self.phase_left = max(cfg.min_warning_s, cfg.warning_s - self.elapsed / 30)
                # Keep at least two safe cells, including a safe cell beside the player.
                safe = set()
                for cell in occupied:
                    row, col = divmod(cell, cfg.cols)
                    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                        if 0 <= row + dr < cfg.rows and 0 <= col + dc < cfg.cols:
                            neighbor = (row + dr) * cfg.cols + col + dc
                            if neighbor not in occupied:
                                safe.add(neighbor)
                                break
                candidates = [cell for cell in range(cfg.rows * cfg.cols) if cell not in safe]
                count = min(len(candidates), 2 + self.round // 3)
                self.cells = self.rng.sample(candidates, count)
            elif self.phase == "warning":
                self.phase, self.phase_left = "lava", cfg.lava_s
            else:
                self.phase, self.phase_left, self.cells = "gap", cfg.gap_s, []
        if self.phase == "lava" and any(cell in self.cells for cell in occupied):
            if self.elapsed - self.last_hit >= 1.0:
                self.lives -= 1
                self.last_hit = self.elapsed
                if self.lives <= 0:
                    self.status = "lost"

    def snapshot(self):
        from dataclasses import asdict

        return {
            "status": self.status,
            "phase": self.phase,
            "lives": self.lives,
            "remaining_s": round(self.remaining, 2),
            "phase_left_s": round(self.phase_left, 2),
            "cells": self.cells,
            "round": self.round,
            "reason": self.reason,
            "config": asdict(self.config),
        }
