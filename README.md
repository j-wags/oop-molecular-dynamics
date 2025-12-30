# Object-Oriented Programming for Molecular Dynamics

This is intended to provide a quick overview of object-oriented programming (OOP) concepts by showing them applied in the context of molecular dynamics.
This example codebase largely mimics the structure of OpenMM, although we've simplified it for clarity.

## Object Hierarchy

Composition structure (concrete classes omitted):

```
Simulation
|-- System
|   |-- ForceField
|-- State
|-- Integrator
```

This diagram shows inheritance relationships; composition relationships are noted inline.

```
System

ForceField (Protocol)
|-- TwoGaussianWellsForceField
|-- HarmonicOscillatorForceField

Integrator
|-- EulerIntegrator
|-- VelocityVerletIntegrator
|-- LangevinIntegratorBase
    |-- LangevinBAOABIntegrator
    |-- VVVRIntegrator

State

Simulation (composes System + Integrator + State)
```

## Concepts

### Encapsulation
- `System` bundles masses, inverse masses, and force field in `state.py`
- `State` keeps positions/velocities together in `state.py`
- Each integrator hides its update steps in `integrators.py`
- Force fields keep potential/force details internal in `force_field.py`

### Abstraction
- `Integrator` defines the shared `step()` interface in `integrators.py`
- `ForceField` protocol defines `compute_forces`/`potential_energy` in `force_field.py`

### Inheritance
- `EulerIntegrator`/`VelocityVerletIntegrator` inherit from `Integrator` in `integrators.py`
- `LangevinBAOABIntegrator`/`VVVRIntegrator` inherit from `LangevinIntegratorBase` in `integrators.py`

### Polymorphism
- `System.forcefield` accepts any `ForceField` implementation in `state.py`
- Any `Integrator` subclass can be used in a `Simulation`

### Composition
- `System` holds a `ForceField` instance in `state.py`
- Integrators operate on `System` + `State` rather than owning them in `integrators.py`
- `Simulation` holds `System`, `Integrator`, and `State` in `simulation.py`

### Single Responsibility
- Force field classes only define potential/forces in `force_field.py`
- Integrators only update time evolution in `integrators.py`
- `usage_example.py` focuses on orchestration + plotting
