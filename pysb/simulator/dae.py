from __future__ import absolute_import
from .base import SimulationResult, Simulator
from pydas.dassl import DASSL
import numpy as np
import sympy
import itertools
import pysb.bng


class DaeSimulator(Simulator):
    """
    Differential Algebraic Equation simulator using DASSL
    """
    def __init__(self, model, tspan=None, initials=None,
                 param_values=None, verbose=False, **kwargs):
        super(DaeSimulator, self).__init__(
            model=model, tspan=tspan, initials=initials,
            param_values=param_values, verbose=verbose, **kwargs)

        pysb.bng.generate_equations(self._model, verbose=self.verbose)

        self._eqn_subs = {e: e.expand_expr(expand_observables=True) for
                          e in self._model.expressions}
        n_species = len(self.model.species)
        dydt = sympy.symbols(','.join('__dydt{}'.format(i)
                                      for i in range(n_species)))
        dae_mat = sympy.Matrix([(self.model.odes[i] - dydt[i]) for i in
                                range(n_species)]).subs(self._eqn_subs)

        self._symbols = sympy.symbols(','.join('__s%d' % sp_id for sp_id in
                                               range(n_species))
                                      + ',') + tuple(model.parameters) + dydt

        self._code_eqs_py = sympy.lambdify(self._symbols,
                                           sympy.flatten(dae_mat))

        self._atol = kwargs.get('atol', 1e-16)
        self._rtol = kwargs.get('rtol', 1e-8)
        self._pdmodel = PyDasModel(self._code_eqs_py)

    def run(self, tspan=None, initials=None, param_values=None):
        tout_all = []
        yout_all = []

        nsims = len(self.param_values)

        for n in range(nsims):
            self._pdmodel.param_vec = self.param_values[n]
            tout = []
            yout = []
            t0 = self.tspan[0]
            y0 = self.initials[n]
            dydt0 = -self._pdmodel.residual(
                t0, y0, np.zeros(len(y0), np.float64)
            )[0]
            self._pdmodel.initialize(
                t0, y0, dydt0, atol=self._atol, rtol=self._rtol
            )
            for t in self.tspan[1:]:
                self._pdmodel.advance(t)
                tout.append(self._pdmodel.t)
                yout.append(self._pdmodel.y.copy())

            tout_all.append(tout)
            yout_all.append(np.asarray(yout))

        return SimulationResult(self, tout_all, yout_all)


class PyDasModel(DASSL):
    def __init__(self, code_eqs_py):
        self._code_eqs_py = code_eqs_py
        self.param_vec = None

    def residual(self, t, y, dydt):
        return np.array(self._code_eqs_py(*itertools.chain(
            y, self.param_vec, dydt))), 0
