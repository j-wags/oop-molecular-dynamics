from dataclasses import dataclass

from integrators import Integrator
from state import State, System


@dataclass
class Simulation:
    """Coordinates a system and an integrator."""

    system: System
    integrator: Integrator
    state: State

    def run_step(self) -> None:
        self.state = self.integrator.step(self.system, self.state)
