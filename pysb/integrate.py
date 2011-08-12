import pysb.bng
import numpy
from scipy.integrate import ode
from scipy.weave import inline
import distutils.errors
import sympy
import re


use_inline = False
# try to inline a C statement to see if inline is functional
try:
    inline('int i;', force=1)
    use_inline = True
except distutils.errors.CompileError as e:
    pass


def odesolve(model, t):
    pysb.bng.generate_equations(model)
    
    param_subs = dict([ (p.name, p.value) for p in model.parameters ])
    param_values = numpy.array([param_subs[p.name] for p in model.parameters])
    param_indices = dict( (p.name, i) for i, p in enumerate(model.parameters) )

    code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
    code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
    for i, p in enumerate(model.parameters):
        code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

    # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
    # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
    # be valid Python.  If the equations ever have more complex things in them, this might fail.
    if not use_inline:
        code_eqs_py = compile(code_eqs, '<%s odes>' % model.name, 'exec')

    y0 = numpy.zeros((len(model.odes),))
    for cp, ic_param in model.initial_conditions:
        si = model.get_species_index(cp)
        y0[si] = ic_param.value

    def rhs(t, y, p):
        ydot = numpy.empty_like(y)
        # note that code_eqs sets ydot as a side effect
        if use_inline:
            inline(code_eqs, ['ydot', 't', 'y', 'p']);
        else:
            exec code_eqs_py in locals()
        return ydot

    nspecies = len(model.species)
    obs_names = [name for name, rp in model.observable_patterns]
    rec_names = ['__s%d' % i for i in range(nspecies)] + obs_names
    yout = numpy.ndarray((len(t), len(rec_names)))

    # perform the actual integration
    integrator = ode(rhs).set_integrator('vode', method='bdf', with_jacobian=True)
    integrator.set_initial_value(y0, t[0]).set_f_params(param_values)
    yout[0, :nspecies] = y0  # FIXME: questionable. first computed step not necessarily continuous from y0, is it?
    i = 1
    while integrator.successful() and integrator.t < t[-1]:
        integrator.integrate(t[i])
        yout[i, :nspecies] = integrator.y
        i += 1

    for i, name in enumerate(obs_names):
        factors, species = zip(*model.observable_groups[name])
        yout[:, nspecies + i] = (yout[:, species] * factors).sum(1)

    dtype = zip(rec_names, (yout.dtype,) * len(rec_names))
    yrec = numpy.recarray((yout.shape[0],), dtype=dtype, buf=yout)
    return yrec
