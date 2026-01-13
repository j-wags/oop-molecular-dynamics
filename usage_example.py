import numpy as np
import matplotlib.pyplot as plt

from force_field import TwoGaussianWellsForceField, HarmonicOscillatorForceField
from integrators import (
    EulerIntegrator,
    LangevinBAOABIntegrator,
    LangevinIntegratorBase,
    VelocityVerletIntegrator,
    VVVRIntegrator,
)
from simulation import Simulation
from state import State, System


def _build_system(num_particles: int) -> System:
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    depths = np.array([3.0, 2.0])
    sigmas = np.array([0.6, 0.8])
    #forcefield = TwoGaussianWellsForceField(centers=centers, depths=depths, sigmas=sigmas)
    forcefield = HarmonicOscillatorForceField(center=centers[0],
                                              spring_constant=1e6)
    masses = np.ones(num_particles)
    return System(masses=masses, forcefield=forcefield)


def _build_state(num_particles: int, seed: int = 7) -> State:
    rng = np.random.default_rng(seed)
    positions = rng.normal(0.0, 0.5, size=(num_particles, 2))
    velocities = rng.normal(0.0, 1.0, size=(num_particles, 2))
    return State(positions=positions, velocities=velocities)


def _thermalize_state(system: System, state: State, temperature: float) -> State:
    dof = system.num_particles * 2
    kB = LangevinIntegratorBase.kB
    kinetic = 0.5 * np.sum(system.masses * np.sum(state.velocities ** 2, axis=1))
    current_temp = 2.0 * kinetic / (dof * kB)
    scale = np.sqrt(temperature / current_temp)
    velocities = state.velocities * scale
    return State(positions=state.positions, velocities=velocities)


def run_energy_tracking(num_steps: int = 1000, dt: float = 0.01) -> None:
    system = _build_system(num_particles=8)
    state = _build_state(num_particles=8)

    euler = EulerIntegrator(dt=dt)
    verlet = VelocityVerletIntegrator(dt=dt)

    euler_energies = []
    verlet_energies = []

    euler_sim = Simulation(system=system, integrator=euler, state=state)
    verlet_sim = Simulation(system=system, integrator=verlet, state=state)

    for _ in range(num_steps):
        euler_sim.run_step()
        verlet_sim.run_step()
        euler_energies.append(system.total_energy(euler_sim.state))
        verlet_energies.append(system.total_energy(verlet_sim.state))

    expected_energy = system.total_energy(state)
    steps = np.arange(1, num_steps + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, euler_energies, label="Euler")
    plt.plot(steps, verlet_energies, label="Velocity Verlet")
    plt.axhline(expected_energy, color="k", linestyle="--", label="Expected Energy")
    plt.xlabel("Step")
    plt.ylabel("Total Energy")
    plt.title("Energy Conservation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Euler energy (first/last):", euler_energies[0], euler_energies[-1])
    print("Velocity Verlet energy (first/last):", verlet_energies[0], verlet_energies[-1])


def run_langevin_tracking(num_steps: int = 1000, dt: float = 0.01) -> None:
    system = _build_system(num_particles=8)
    state = _build_state(num_particles=8)

    temperature = 300.0
    state = _thermalize_state(system, state, temperature)
    baoab = LangevinBAOABIntegrator(dt=dt, temperature=temperature, friction_coeff=5.0)
    vvvr = VVVRIntegrator(dt=dt, temperature=temperature, friction_coeff=5.0)

    baoab_temps = []
    vvvr_temps = []
    baoab_speeds = []
    vvvr_speeds = []

    baoab_sim = Simulation(system=system, integrator=baoab, state=state)
    vvvr_sim = Simulation(system=system, integrator=vvvr, state=state)

    dof = system.num_particles * 2
    kB = LangevinIntegratorBase.kB

    for _ in range(num_steps):
        baoab_sim.run_step()
        vvvr_sim.run_step()
        baoab_ke = 0.5 * np.sum(system.masses * np.sum(baoab_sim.state.velocities ** 2, axis=1))
        vvvr_ke = 0.5 * np.sum(system.masses * np.sum(vvvr_sim.state.velocities ** 2, axis=1))
        baoab_temps.append(2.0 * baoab_ke / (dof * kB))
        vvvr_temps.append(2.0 * vvvr_ke / (dof * kB))
        baoab_speeds.extend(np.linalg.norm(baoab_sim.state.velocities, axis=1))
        vvvr_speeds.extend(np.linalg.norm(vvvr_sim.state.velocities, axis=1))

    steps = np.arange(1, num_steps + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, baoab_temps, label="BAOAB")
    plt.plot(steps, vvvr_temps, label="VVVR")
    plt.axhline(temperature, color="k", linestyle="--", label="Target Temperature")
    plt.xlabel("Step")
    plt.ylabel("Instantaneous Temperature (K)")
    plt.title("Langevin Temperature Fluctuations")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("BAOAB temperature (first/last):", baoab_temps[0], baoab_temps[-1])
    print("VVVR temperature (first/last):", vvvr_temps[0], vvvr_temps[-1])

    plt.figure(figsize=(7, 4))
    plt.hist(baoab_speeds, bins=12, alpha=0.6, density=True, label="BAOAB")
    plt.hist(vvvr_speeds, bins=12, alpha=0.6, density=True, label="VVVR")
    max_speed = max(max(baoab_speeds), max(vvvr_speeds))
    speed_grid = np.linspace(0.0, max_speed, 200)
    sigma = np.sqrt(LangevinIntegratorBase.kB * temperature / system.masses)
    expected_pdf = np.zeros_like(speed_grid)
    for sigma_i in sigma:
        expected_pdf += (speed_grid / (sigma_i ** 2)) * np.exp(-speed_grid ** 2 / (2.0 * sigma_i ** 2))
    expected_pdf /= sigma.size
    plt.plot(speed_grid, expected_pdf, color="k", linestyle="--", label="Expected Distribution")
    plt.xlabel("Speed")
    plt.ylabel("Density")
    plt.title("Langevin Speed Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_energy_tracking(10000)
    #run_langevin_tracking()
