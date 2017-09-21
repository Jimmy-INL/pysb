from pysb.simulator.dae import DaeSimulator
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.examples import earm_1_0, robertson
import numpy as np
from pandas.util.testing import assert_frame_equal
from unittest import SkipTest


def run_dae_against_scipy(eqn_mode):
    if eqn_mode == 'weave':
        try:
            import weave
        except ImportError:
            raise SkipTest('Test skipped (weave library not installed)')
    elif eqn_mode == 'cython':
        try:
            import cython
        except ImportError:
            raise SkipTest('Test skipped (cython library not installed)')

    atol = 1e-16
    rtol = 1e-8

    test_simulations = {
        earm_1_0: np.linspace(0, 20000, 101),
        robertson: np.linspace(0, 100, 101)
    }

    for m, tspan in test_simulations.items():
        model = m.model
        dae_results = []

        sim = DaeSimulator(model=model, tspan=tspan, eqn_mode=eqn_mode)
        dae_results.append(sim.run().dataframe)

        odesim = ScipyOdeSimulator(model=model, tspan=tspan,
                                   atol=atol, rtol=rtol,
                                   integrator='lsoda')
        ode_df = odesim.run().dataframe

        assert_frame_equal(dae_results[0], ode_df, check_less_precise=True)


def test_dae_against_scipy():
    for eqn_mode in ['python', 'cython', 'weave']:
        yield(run_dae_against_scipy, eqn_mode)


def test_dae_multi_initials():
    A = robertson.model.monomers['A']
    sim = DaeSimulator(model=robertson.model, tspan=np.linspace(0, 100, 101),
                       eqn_mode='python')
    obs = sim.run(initials={A(): [0, 1, 2]}).observables

    assert np.allclose([obs[i]['A_total'][0] for i in range(3)], [0, 1, 2])


def test_dae_multi_params():
    sim = DaeSimulator(model=robertson.model, tspan=np.linspace(0, 100, 101),
                       eqn_mode='python')
    obs = sim.run(param_values={'k1': [0, 0.04]}).observables

    print([obs[i]['B_total'][-1] for i in range(2)])

    assert np.allclose([obs[i]['B_total'][-1] for i in range(2)],
                       [0, 6.15359e-06])
