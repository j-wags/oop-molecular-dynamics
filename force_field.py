from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ForceField(Protocol):
    """Protocol for simple force fields."""

    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute forces for the given positions."""
        ...

    def potential_energy(self, positions: np.ndarray) -> float:
        """Compute total potential energy for the given positions."""
        ...


@dataclass
class TwoGaussianWellsForceField:
    """2D surface with two Gaussian wells."""

    centers: np.ndarray
    depths: np.ndarray
    sigmas: np.ndarray

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=float)
        self.depths = np.asarray(self.depths, dtype=float)
        self.sigmas = np.asarray(self.sigmas, dtype=float)
        if self.centers.shape != (2, 2):
            raise ValueError("centers must have shape (2, 2)")
        if self.depths.shape != (2,):
            raise ValueError("depths must have shape (2,)")
        if self.sigmas.shape != (2,):
            raise ValueError("sigmas must have shape (2,)")
        if np.any(self.sigmas <= 0.0):
            raise ValueError("sigmas must be positive")

    def _as_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            if positions.shape[0] != 2:
                raise ValueError("positions must have length 2 for 2D points")
            positions = positions[np.newaxis, :]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (num_particles, 2)")
        return positions

    def potential_energy(self, positions: np.ndarray) -> float:
        positions = self._as_positions(positions)
        deltas = positions[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        r2 = np.sum(deltas ** 2, axis=-1)
        exponents = np.exp(-r2 / (2.0 * self.sigmas ** 2))
        per_particle = -np.sum(self.depths * exponents, axis=1)
        return float(np.sum(per_particle))

    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        positions = self._as_positions(positions)
        deltas = positions[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        r2 = np.sum(deltas ** 2, axis=-1)
        exponents = np.exp(-r2 / (2.0 * self.sigmas ** 2))
        prefactor = -(self.depths / (self.sigmas ** 2))[np.newaxis, :, np.newaxis]
        forces = np.sum(prefactor * exponents[:, :, np.newaxis] * deltas, axis=1)
        return forces


@dataclass
class HarmonicOscillatorForceField:
    """2D harmonic oscillator centered at a point."""

    center: np.ndarray
    spring_constant: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError("center must have shape (2,)")
        if self.spring_constant <= 0.0:
            raise ValueError("spring_constant must be positive")

    def _as_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            if positions.shape[0] != 2:
                raise ValueError("positions must have length 2 for 2D points")
            positions = positions[np.newaxis, :]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (num_particles, 2)")
        return positions

    def potential_energy(self, positions: np.ndarray) -> float:
        positions = self._as_positions(positions)
        deltas = positions - self.center[np.newaxis, :]
        r2 = np.sum(deltas ** 2, axis=1)
        return float(0.5 * self.spring_constant * np.sum(r2))

    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        positions = self._as_positions(positions)
        deltas = positions - self.center[np.newaxis, :]
        return -self.spring_constant * deltas
