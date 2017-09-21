from __future__ import absolute_import
from .base import SimulationResult, Simulator
from pydas.dassl import DASSL
import numpy as np
import sympy
import itertools
import pysb.bng
import re
try:
    import cython
except ImportError:
    cython = None
try:
    import weave
except ImportError:
    weave = None


class DaeSimulator(Simulator):
    """
    Differential Algebraic Equation simulator using PyDAS and DASSL

    .. warning::
        The interface for this class is considered experimental and may
        change without warning as PySB is updated.

    This Simulator integrates PySB with the Fortran-based differential
    algebraic system solver [DASSL]_. It requires the [PyDAS]_ library to be
    installed - the easiest way to do this is with [Anaconda]_::

        conda install -c rmg -y pydas

    See the [PyDAS]_ README for more detailed installation instructions.

    .. [DASSL] https://www.osti.gov/scitech/servlets/purl/5882821 (PDF)
    .. [PyDAS] https://github.com/ReactionMechanismGenerator/PyDAS
    .. [Anaconda] https://www.anaconda.com/download/

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to
        satisfaction of a logical stopping criterion (see 'tout' below).
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.
    **kwargs : dict
        Extra keyword arguments, including:

        * ``eqn_mode``: Equation evaluation mode - one of `weave`, `cython`
        or `python`. They should give identical results, differing only in
        speed. `weave` uses the `weave` library to compile the system of
        equations into C code. `cython` does likewise with the `cython`
        library (`weave` is only available on Python 2, but seems to be
        faster). `python` uses `sympy`'s `lambdify` function to execute the
        system of equations in pure Python, which is the most compatible but
        slowest option. The `python` option might be necessary if your model
        contains non-standard `sympy` functions for example. In general,
        we recommend `weave` for Python 2 and `cython` for Python 3 in the
        first instance.

        * ``atol``: Absolute tolerance (default: 1e-16)

        * ``rtol``: Relative tolerance (default: 1e-8)

    Notes
    -----
    If ``tspan`` is not defined, it may be defined in the call to the
    ``run`` method.

    Examples
    --------
    Simulate a model and display the results for an observable:

    >>> from pysb.examples.robertson import model
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> sim = DaeSimulator(model, tspan=np.linspace(0, 40, 10), \
                           eqn_mode='python')
    >>> simulation_result = sim.run()
    >>> print(simulation_result.observables['A_total']) \
        #doctest: +NORMALIZE_WHITESPACE
    [ 1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    For further information on retrieving trajectories (species,
    observables, expressions over time) from the ``simulation_result``
    object returned by :func:`run`, see the examples under the
    :class:`SimulationResult` class.

    """
    _supports = {'multi_initials': True,
                 'multi_param_values': True}

    def __init__(self, model, tspan=None, initials=None,
                 param_values=None, verbose=False, eqn_mode='weave', **kwargs):
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

        self._atol = kwargs.get('atol', 1e-16)
        self._rtol = kwargs.get('rtol', 1e-8)

        if eqn_mode == 'weave':
            if weave is None:
                raise ImportError('"weave" library not installed. Install '
                                  'weave or switch eqn_mode to "python" or '
                                  '"cython"')
            code_eqs = '\n'.join(['delta[%d] = %s;' %
                                  (i, sympy.ccode(o))
                                  for i, o in enumerate(dae_mat)])
            code_eqs = self._eqn_substitutions(code_eqs)
            self._pdmodel = PyDasModelWeave(code_eqs, n_species)
        elif eqn_mode == 'cython':
            if cython is None:
                raise ImportError('"cython" library not installed. Install '
                                  'cython or switch eqn_mode to "python" or '
                                  '"weave"')
            code_eqs = '\n'.join(['delta[%d] = %s' %
                                  (i, sympy.ccode(o))
                                  for i, o in enumerate(dae_mat)])
            code_eqs = self._eqn_substitutions(code_eqs)
            code_eqs += '\nreturn delta'
            code_eqs = 'cdef double delta[{}]\n'.format(len(dae_mat)) + \
                       code_eqs

            self._pdmodel = PyDasModelCython(code_eqs)
        else:
            self._symbols = sympy.symbols(','.join('__s%d' % sp_id for sp_id in
                                                   range(n_species))
                                          + ',') + tuple(model.parameters) + dydt

            code_eqs_py = sympy.lambdify(self._symbols,
                                         sympy.flatten(dae_mat))
            self._pdmodel = PyDasModel(code_eqs_py)

    def _eqn_substitutions(self, eqns):
        """String substitutions on the sympy C code for the ODE RHS and
        Jacobian functions to use appropriate terms for variables and
        parameters."""
        # Substitute 'y[i]' for '__si'
        eqns = re.sub(r'\b__s(\d+)\b',
                      lambda m: 'y[%s]' % (int(m.group(1))),
                      eqns)

        # Substitute 'dydt[i]' for '__dydti'
        eqns = re.sub(r'\b__dydt(\d+)\b',
                      lambda m: 'dydt[%s]' % (int(m.group(1))),
                      eqns)

        # Substitute 'p[i]' for any named parameters
        for i, p in enumerate(self._model.parameters):
            eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, eqns)
        return eqns

    def run(self, tspan=None, initials=None, param_values=None):
        super(DaeSimulator, self).run(tspan=tspan,
                                      initials=initials,
                                      param_values=param_values)
        tout_all = []

        n_sims = len(self.param_values)

        trajectories = np.ndarray((n_sims,
                                   len(self.tspan),
                                   len(self._model.species)))

        for n in range(n_sims):
            self._pdmodel.param_vec = self.param_values[n]
            t0 = self.tspan[0]
            tout = [t0]
            y0 = self.initials[n]
            trajectories[n][0] = y0
            dydt0 = -self._pdmodel.residual(
                t0, y0, np.zeros(len(y0), np.float64)
            )[0]
            self._pdmodel.initialize(
                t0, y0, dydt0, atol=self._atol, rtol=self._rtol
            )
            for t_ind in range(len(self.tspan) - 1):
                self._pdmodel.advance(self.tspan[t_ind + 1])
                tout.append(self._pdmodel.t)
                trajectories[n][t_ind + 1] = self._pdmodel.y

            tout_all.append(tout)

        return SimulationResult(self, tout_all, trajectories)


class PyDasModelWeave(DASSL):
    def __init__(self, code_eqs, num_odes):
        self._code_eqs = code_eqs
        self.param_vec = None
        self.num_odes = num_odes

    def residual(self, t, y, dydt):
        p = self.param_vec
        delta = np.empty(self.num_odes)
        weave.inline(self._code_eqs, ['delta', 't', 'y', 'dydt', 'p'])
        return np.array(delta), 0


class PyDasModelCython(DASSL):
    def __init__(self, code_eqs):
        self._code_eqs = code_eqs
        self.param_vec = None

    def residual(self, t, y, dydt):
        return np.array(cython.inline(self._code_eqs, p=self.param_vec)), 0


class PyDasModel(DASSL):
    def __init__(self, code_eqs):
        self._code_eqs = code_eqs
        self.param_vec = None

    def residual(self, t, y, dydt):
        return np.array(self._code_eqs(*itertools.chain(
            y, self.param_vec, dydt))), 0
