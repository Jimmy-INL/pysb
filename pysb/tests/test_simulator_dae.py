from pysb.simulator.dae import DaeSimulator
from pysb.examples import earm_1_0
import time
import numpy as np


def test_dae():
    model = earm_1_0.model
    tspan = np.linspace(0, 20000, 20001)
    atol = 1e-16
    rtol = 1e-8

    pds = DaeSimulator(model=model, tspan=tspan)
    start = time.clock()
    pds.run()
    print('DASSL: {}ms'.format((time.clock() - start) * 1000))

    from pysb.simulator.scipyode import ScipyOdeSimulator

    for weave_inline in (True, False):
        ScipyOdeSimulator._use_inline = weave_inline
        odesim = ScipyOdeSimulator(model=model, tspan=tspan,
                                   atol=atol, rtol=rtol,
                                   integrator='lsoda')
        start = time.clock()
        odesim.run()
        print('LSODA (inline {}): {}ms'.format(
            weave_inline,
            (time.clock() - start) * 1000))

        odesim = ScipyOdeSimulator(model=model, tspan=tspan,
                                   atol=atol, rtol=rtol,
                                   integrator='vode')
        start = time.clock()
        odesim.run()
        print('VODE (inline {}): {}ms'.format(
            weave_inline,
            (time.clock() - start) * 1000))


    # print((pds.run().dataframe - odesim.run().dataframe).abs().max())

    raise
