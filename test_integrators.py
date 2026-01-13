from integrators import VelocityVerletIntegrator

def test_verlet_name():
    vi = VelocityVerletIntegrator(1)
    assert vi.this_integrators_name == "Verlet Integrator"
