from dataclasses import dataclass, field

import numpy as np

from force_field import ForceField


@dataclass
class State:
    """State of our toy MD system"""

    positions: np.ndarray
    velocities: np.ndarray


@dataclass
class System:
    """System bundles masses with a force field used by integrators."""

    masses: np.ndarray
    forcefield: ForceField
    inv_masses: np.ndarray = field(init=False)
    num_particles: int = field(init=False)

    def __post_init__(self):
        self.masses = np.asarray(self.masses, dtype=float)
        if self.masses.ndim != 1:
            raise ValueError("masses must be a 1D array of shape (num_particles,)")
        self.inv_masses = 1.0 / self.masses
        self.num_particles = self.masses.shape[0]

    def total_energy(self, state: State) -> float:
        kinetic = 0.5 * np.sum(self.masses * np.sum(state.velocities ** 2, axis=1))
        potential = self.forcefield.potential_energy(state.positions)
        return float(kinetic + potential)
