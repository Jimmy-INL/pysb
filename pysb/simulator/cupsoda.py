from __future__ import print_function as _
from pysb.simulator.base import Simulator, SimulatorException, SimulationResult
import pysb.bng
import numpy as np
from scipy.constants import N_A
import os
import re
import subprocess
import tempfile
import time
import warnings
import shutil
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    use_pycuda = True
except ImportError:
    use_pycuda = False
    pass

_cupsoda_path = None

def set_cupsoda_path(directory):
    global _cupsoda_path
    _cupsoda_path = os.path.join(directory, 'cupSODA')
    # Make sure file exists and that it is not a directory
    if not os.access(_cupsoda_path, os.F_OK) or not \
            os.path.isfile(_cupsoda_path):
        raise SimulatorException(
                        'Could not find cupSODA binary in ' +
                        os.path.abspath(directory) + '.')
    # Make sure file has executable permissions
    elif not os.access(_cupsoda_path, os.X_OK):
        raise SimulatorException(
                        "cupSODA binary in " + os.path.abspath(directory) +
                        " does not have executable permissions.")


def _get_cupsoda_path():
    """
    Return the path to the cupSODA executable.

    Looks for the cupSODA executable in a user-defined location set via
    ``set_cupsoda_path``, the environment variable CUPSODAPATH or in a few
    hard-coded standard locations.
    """
    global _cupsoda_path

    # Just return cached value if it's available
    if _cupsoda_path:
        return _cupsoda_path

    path_var = 'CUPSODAPATH'
    bin_dirs = [
        '/usr/local/share/cupSODA',
        'c:/Program Files/cupSODA',
    ]

    def _check_bin_dir(bin_dir):
        # Return the full path to the cupSODA executable or False if it
        # can't be found in one of the expected places.
        bin_path = os.path.join(bin_dir, 'cupSODA')
        if os.access(bin_path, os.F_OK):
            return bin_path
        else:
            return False

    # First check the environment variable, which has the highest precedence
    if path_var in os.environ:
        bin_path = _check_bin_dir(os.environ[path_var])
        if not bin_path:
            raise SimulatorException(
                            'Environment variable %s is set but the path could'
                            ' not be found, is not accessible or does not '
                            'contain a cupSODA executable file.' % path_var)
    # If the environment variable isn't set, check the standard locations
    # Check the standard locations for the executable
    else:
        for b in bin_dirs:
            bin_path = _check_bin_dir(b)
            if bin_path:
                break
            else:
                raise SimulatorException(
                            'Could not find cupSODA installed in one of '
                            'the following locations:' +
                            ''.join('\n    ' + x for x in bin_dirs) +
                            '\nPlease put the executable (or a link to '
                            'it) in one of these locations or set the '
                            'path using set_cupsoda_path().')

    # Cache path for future use
    _cupsoda_path = bin_path
    return bin_path


class CupSodaSimulator(Simulator):
    """An interface for running cupSODA, a CUDA implementation of LSODA.
    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like, optional
        Time values at which the integrations are sampled. The first and last
        values define the time range.
    initials : list-like, optional
        Initial species concentrations for all simulations. Dimensions are 
        N_SIMS x number of species.
    param_values : list-like, optional
        Parameters for all simulations. Dimensions are N_SIMS x number of 
        parameters.
    verbose : bool, optional
        Verbose output
    **kwargs: dict
        Extra keyword arguments, including:
        * ``gpu``: Index of GPU to run on (default: 0)
        * ``vol``: System volume; required if model encoded in extrinsic 
          (number) units (default: None)
        * ``obs_species_only``: Only output species contained in observables
          (default: True) 
        * ``cleanup``: Delete all temporary files after the simulation is 
          finished. Includes both BioNetGen and cupSODA files. Useful for 
          debugging (default: True)
        * ``prefix``: Prefix for the temporary directory containing cupSODA input 
          and output files (default: model name)
        * ``base_dir``: Directory in which temporary directory with cupSODA input  
          and output files are placed (default: system directory determined by 
          `tempfile.mkdtemp`)
        * ``integrator``: Name of the integrator to use; see 
          `default_integrator_options` (default: 'cupsoda')
        * ``integrator_options``: A dictionary of keyword arguments to
          supply to the integrator; see `default_integrator_options`.

    Attributes
    ----------
    verbose: bool
        Verbosity setting.
    model : pysb.Model
        Model passed to the constructor.
    tspan : numpy.ndarray
        Time values passed to the constructor.
    tout: numpy.ndarray
        Time points returned by the simulator (may be different from ``tspan``
        if simulation is interrupted for some reason).
    initials : numpy.ndarray
        Initial species concentrations for all simulations. Dimensions are 
        number of simulations x number of species.
    param_values : numpy.ndarray
        Parameters for all simulations. Dimensions are number of simulations 
        x number of parameters.
    gpu : int
        Index of GPU being run on
    vol : float or None
        System volume
    outdir : string
        Temporary directory where cupSODA output files are placed. Input
        files are also placed here.
    opts: dict
        Dictionary of options for the integrator in use.
    integrator : string
        Name of the integrator in use.
    default_integrator_options : dict
        Nested dictionary of default options for all supported integrators.   
        
    Notes
    -----
    1. If `vol` is defined, species amounts and rate constants are assumed
       to be in number units and are automatically converted to concentration
       units before generating the cupSODA input files. The species
       concentrations returned by cupSODA are converted back to number units
       during loading.

    2. If `obs_species_only` is True, only the species contained within 
       observables are output by cupSODA. All other concentrations are set 
       to 'nan'.
    """

    _supports = { 'multi_initials' : True,
                  'multi_param_values' : True }

    _memory_options = {'global': '0', 'shared': '1', 'sharedconstant': '2'}

    default_integrator_options = {
    # some sane default options for a few well-known integrators
        'cupsoda': {
            'max_steps': 20000,  # max no. of internal iterations (LSODA's MXSTEP)
            'atol': 1e-8,  # absolute tolerance
            'rtol': 1e-8,  # relative tolerance
            'n_blocks': None,  # number of GPU blocks
            'memory_usage': 'sharedconstant'  # see _memory_options dict
        },
    }
    
    def __init__(self, model, tspan=None, initials=None, param_values=None,
                 verbose=False, **kwargs):
        super(CupSodaSimulator, self).__init__(model, 
                                               tspan=tspan,
                                               initials=initials,
                                               param_values=param_values,
                                               verbose=verbose, 
                                               **kwargs)
        self.gpu = kwargs.get('gpu', 0)
        self.vol = kwargs.get('vol', None)
        self._obs_species_only = kwargs.get('obs_species_only', True)
        self._cleanup = kwargs.get('cleanup', True)
        self._prefix = kwargs.get('prefix', self._model.name.replace('.','_'))
        self._base_dir = kwargs.get('base_dir', None)
        self.integrator = kwargs.get('integrator', 'cupsoda')
        
        # generate the equations for the model
        pysb.bng.generate_equations(self._model, self._cleanup, self.verbose)
                
        # build integrator options list from our defaults and any kwargs
        # passed to this function
        options = {}
        if self.default_integrator_options.get(self.integrator):
            options.update(
                self.default_integrator_options[self.integrator])  # default options
        else:
            raise SimulatorException(
                "Integrator type '" + integrator + "' not recognized.")
        options.update(kwargs.get('integrator_options', {}))  # overwrite
        
        # defaults
        self.opts = options
        self._out_species = None
        
        # private variables (to reduce the number of function calls)
        self._len_tspan = len(self.tspan)
        self._len_rxns = len(self._model.reactions)
        self._len_species = len(self._model.species)
        self._len_params = len(self._model.parameters)
        self._model_parameters_rules = self._model.parameters_rules()
        
    def run(self, tspan=None, initials=None, param_values=None):
        """Perform a set of integrations.

        Returns a :class:`.SimulationResult` object.

        Parameters
        ----------
        tspan : list-like, optional
            Time values at which the integrations are sampled. The first and last
            values define the time range.
        initials : list-like, optional
            Initial species concentrations for all simulations. Dimensions are
            number of simulation x number of species.    
        param_values : list-like, optional
            Parameters for all simulations. Dimensions are number of simulations x
            number of parameters.

        Returns
        -------
        A :class:`SimulationResult` object

        Notes
        -----
        1. An exception is thrown if `tspan` is not defined in either `__init__`
           or `run`.
           
        2. If neither `initials` nor `param_values` are defined in either 
           `__init__` or `run` a single simulation is run with the initial 
           concentrations and parameter values defined in the model.
        """
        super(CupSodaSimulator, self).run(tspan=tspan,
                                           initials=initials,
                                           param_values=param_values)

        # Create directories for cupSODA input and output files
        self.outdir = tempfile.mkdtemp(prefix=self._prefix+'_', dir=self._base_dir)
        if self.verbose:
            print("Output directory is %s" % self.outdir)
        self._cupsoda_infiles_dir = os.path.join(self.outdir,"INPUT")
        os.mkdir(self._cupsoda_infiles_dir)
        self._cupsoda_outfiles_dir = os.path.join(self.outdir,"OUTPUT")
        os.mkdir(self._cupsoda_outfiles_dir)

        # Path to cupSODA executable
        bin_path = _get_cupsoda_path()

        # Number of blocks
        n_blocks = self._get_nblocks(self.gpu)
        
        # Create cupSODA input files
        self._create_input_files(self._cupsoda_infiles_dir)

        # Build command
        # ./cupSODA input_model_folder blocks output_folder simulation_
        # file_prefix gpu_number fitness_calculation memory_use dump
        command = [bin_path, 
                   self._cupsoda_infiles_dir, 
                   str(n_blocks),
                   self._cupsoda_outfiles_dir, 
                   self._prefix,
                   str(self.gpu),
                   '0', 
                   self._memory_usage,
                   '1' if self.verbose else '0']
        print("Running cupSODA: " + ' '.join(command))

        # Run simulation and return trajectories
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        if self.verbose:
            for line in iter(p.stdout.readline, b''):
                print(line, end="")
        (p_out, p_err) = p.communicate()
        if p.returncode:
            raise SimulatorException( p_out.rstrip("at line") +
                                     "\n" + p_err.rstrip() )
        self.tout, trajectories = self._load_trajectories(
                                       self._cupsoda_outfiles_dir)
        if self._cleanup:
            shutil.rmtree(self.outdir)
        
        return SimulationResult(self, trajectories)

    @property
    def _memory_usage(self):
        try:
            return self._memory_options[self.opts['memory_usage']]
        except KeyError:
            raise Exception('memory_usage must be one of %s',
                            self._memory_options.keys())

    def _get_nblocks(self, gpu):
        n_blocks = self.opts.get('n_blocks')
        if n_blocks is None:
            default_threads_per_block = 32
            bytes_per_float = 4
            memory_per_thread = (self._len_species + 1) * bytes_per_float
            if not use_pycuda:
                threads_per_block = default_threads_per_block
            else:
                device = cuda.Device(gpu)
                attrs = device.get_attributes()
                shared_memory_per_block = attrs[pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
                upper_limit_threads_per_block = attrs[pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK]
                max_threads_per_block = min(shared_memory_per_block / memory_per_thread,
                                            upper_limit_threads_per_block)
                threads_per_block = min(max_threads_per_block, default_threads_per_block)
            n_blocks = int(np.ceil(1. * len(self.param_values) / threads_per_block))
        return n_blocks

    def _create_input_files(self, dir):

        n_sims = len(self.param_values)

        # atol_vector
        with open(os.path.join(dir, "atol_vector"),
                  'wb') as atol_vector:
            for i in range(self._len_species):
                atol_vector.write(str(self.opts.get('atol')))
                if i < self._len_species - 1:
                    atol_vector.write("\n")

        # c_matrix
        with open(os.path.join(dir, "c_matrix"), 'wb') as c_matrix:
            cmtx = self._get_cmatrix()
            for i in range(n_sims):
                line = ""
                for j in range(self._len_rxns):
                    if j > 0:
                        line += "\t"
                    line += str(cmtx[i][j])
                c_matrix.write(line)
                if i < n_sims - 1:
                    c_matrix.write("\n")

        # cs_vector
        with open(os.path.join(dir, "cs_vector"), 'wb') as cs_vector:
            self._out_species = range(self._len_species) # species to output
            if self._obs_species_only:
                self._out_species = [False for sp in self._model.species]
                for obs in self._model.observables:
                    for i in obs.species:
                        self._out_species[i] = True
                self._out_species = [i for i in range(self._len_species) if
                               self._out_species[i] is True]
            for i in range(len(self._out_species)):
                if i > 0:
                    cs_vector.write("\n")
                cs_vector.write(str(self._out_species[i]))

        # left_side
        with open(os.path.join(dir, "left_side"), 'wb') as left_side:
            for i in range(self._len_rxns):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    stoich = 0
                    for k in self._model.reactions[i]['reactants']:
                        if j == k:
                            stoich += 1
                    line += str(stoich)
                if i < self._len_rxns - 1:
                    left_side.write(line + "\n")
                else:
                    left_side.write(line)

        # max_steps
        with open(os.path.join(dir, "max_steps"), 'wb') as mxsteps:
            mxsteps.write(str(self.opts['max_steps']))

        # model_kind
        with open(os.path.join(dir, "modelkind"),
                  'wb') as model_kind:
            # always set modelkind to 'deterministic'
            model_kind.write("deterministic")

        # MX_0
        with open(os.path.join(dir, "MX_0"), 'wb') as MX_0:
            mx0 = self.initials
            # if a volume has been defined, rescale populations
            # by N_A*vol to get concentration
            # (NOTE: act on a copy of self.initials, not
            # the original, which we don't want to modify)
            if self.vol:
                mx0 = mx0.copy()
                mx0 /= (N_A * self.vol)
                # Set the concentration of __source() to 1
                for i,sp in enumerate(self._model.species):
                    if str(sp) == '__source()':
                        mx0[:, i] = 1.
                        break
            for i in range(n_sims):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    line += str(mx0[i][j])
                MX_0.write(line)
                if i < n_sims - 1:
                    MX_0.write("\n")

        # right_side
        with open(os.path.join(dir, "right_side"),
                  'wb') as right_side:
            for i in range(self._len_rxns):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    stochiometry = 0
                    for k in self._model.reactions[i]['products']:
                        if j == k:
                            stochiometry += 1
                    line += str(stochiometry)
                if i < self._len_rxns - 1:
                    right_side.write(line + "\n")
                else:
                    right_side.write(line)

        # rtol
        with open(os.path.join(dir, "rtol"), 'wb') as rtol:
            rtol.write(str(self.opts.get('rtol')))

        # t_vector
        with open(os.path.join(dir, "t_vector"), 'wb') as t_vector:
            for t in self.tspan:
                t_vector.write(str(float(t)) + "\n")

        # time_max
        with open(os.path.join(dir, "time_max"), 'wb') as time_max:
            time_max.write(str(float(self.tspan[-1])))

    def _get_cmatrix(self):
        if self.verbose:
            print("Constructing the c_matrix:")
        c_matrix = np.zeros((len(self.param_values), self._len_rxns))
        par_names = [p.name for p in self._model_parameters_rules]
        rate_mask = np.array([p in self._model_parameters_rules for p in
                              self._model.parameters])
        par_dict = {par_names[i]: i for i in range(len(par_names))}
        rate_args = []
        par_vals = self.param_values[:, rate_mask]
        rate_order = []
        for rxn in self._model.reactions:
            rate_args.append([arg for arg in rxn['rate'].args if
                              not re.match("_*s", str(arg))])
            reactants = 0
            for i in rxn['reactants']:
                if not str(self._model.species[i]) == '__source()':
                    reactants += 1
            rate_order.append(reactants)
        if self.verbose:
            output = 0.01 * len(par_vals)
            output = int(output) if output > 1 else 1
        for i in range(len(par_vals)):
            if self.verbose and i % output == 0:
                print(str(int(round(100. * i / len(par_vals)))) + "%")
            for j in range(self._len_rxns):
                rate = 1.0
                for r in rate_args[j]:
                    x = str(r)
                    if x in par_names:
                        rate *= par_vals[i][par_dict[x]]
                    else:
                        # FIXME: need to detect non-numbers and throw an error
                        rate *= float(x)
                # volume correction
                if self.vol:
                    rate *= (N_A * self.vol) ** (rate_order[j] - 1)
                c_matrix[i][j] = rate
        if self.verbose:
            print("100%")
        return c_matrix

    def _load_trajectories(self, dir):
        """Read simulation results from output files.

        Returns `tout` and `trajectories` arrays.
        """
        files = [filename for filename in os.listdir(dir) if
                 re.match(self._prefix, filename)]
        if len(files) == 0:
            raise SimulatorException("Cannot find any output files to load data from.")
        if len(files) != len(self.param_values):
            raise SimulatorException("Number of input files (%d) does not match number "
                                     "of requested simulations (%d)." % 
                                     (len(files), len(self.param_values)))
        n_sims = len(files)
        trajectories = [None] * n_sims
        tout = [None] * n_sims
        traj_n = np.ones((self._len_tspan, self._len_species)) * float('nan')
        tout_n = np.ones(self._len_tspan) * float('nan')
        # load the data
        indir_prefix = os.path.join(dir, self._prefix)
        for n in range(n_sims):
            trajectories[n] = traj_n.copy()
            tout[n] = tout_n.copy()
            filename = indir_prefix + "_" + str(n)
            if not os.path.isfile(filename):
                raise Exception("Cannot find input file " + filename)
            # determine optimal loading method
            if n == 0:
                (data, use_pandas) = self._test_pandas(filename)
            # load data
            else:
                if use_pandas:
                    data = self._load_with_pandas(filename)
                else:
                    data = self._load_with_openfile(filename)
            # store data
            tout[n] = data[:, 0]
            trajectories[n][:, self._out_species] = data[:, 1:]
            # volume correction
            if self.vol:
                trajectories[n][:, self._out_species] *= (N_A * self.vol)
        return np.array(tout), np.array(trajectories)

    def _test_pandas(self, filename):
        """ calculates the fastest method to load in data
        If the file is large, pandas is generally significantly faster.

        :param filename:
        :return: numpy.ndarray, bool
        """
        # using open(filename,...)
        start = time.time()
        data = self._load_with_openfile(filename)
        end = time.time()
        load_time_openfile = end - start
        
        # using pandas
        if pd:
            start = time.time()
            self._load_with_pandas(filename)
            end = time.time()
            load_time_pandas = end - start
            if load_time_pandas < load_time_openfile:
                return data, True
            
        return data, False

    def _load_with_pandas(self, filename):
        data = pd.read_csv(filename, sep='\t', skiprows=None, header=None).as_matrix()
        return data

    def _load_with_openfile(self, filename):
        with open(filename, 'rb') as f:
            data = [line.rstrip('\n').split() for line in f]
        data = np.array(data, dtype=np.float, copy=False)
        return data
    
def run_cupsoda(model, tspan, initials=None, param_values=None, integrator='cupsoda',
         cleanup=True, verbose=False, **kwargs):
    '''Wrapper method for running cupSODA simulations.
    
    Parameters
    ----------
    See ``CupSodaSimulator`` constructor.
    
    Returns
    -------
    SimulationResult.all : list of record arrays
        List of trajectory sets. The first dimension contains species,
        observables and expressions (in that order)
    '''
    sim = CupSodaSimulator(model, tspan=tspan, integrator=integrator, 
                           cleanup=cleanup, verbose=verbose, **kwargs)    
    simres = sim.run(initials=initials, param_values=param_values)
    return simres.all
