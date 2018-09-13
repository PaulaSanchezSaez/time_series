import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy import optimize, stats
import emcee
import statsmodels.api as sm

def to_log(x, xerr=[]):
    """
    Take linear measurements and uncertainties and transform to log
    values.
    """
    logx = np.log10(np.array(x))
    if np.any(xerr):
        xerr = np.log10(np.array(x)+np.array(xerr)) - logx
    else:
        xerr = np.zeros_like(x)
    return logx, xerr

def mcmc_regression_1x(x1, x2, x1err=None, x2err=None, start=(1.,1.,0.5),
         starting_width=0.01, logify=True,
         nsteps=5000, nwalkers=100, nburn=0, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.
    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter
    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default True)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.
    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.
    """

    # just in case
    x1 = np.array(x1)
    x2 = np.array(x2)
    if x1err is None:
        x1err = np.zeros(x1.size)
    if x2err is None:
        x2err = np.zeros(x1.size)

    def lnlike(theta, x, y, xerr, yerr):
        """Likelihood"""
        a, b, s = theta
        model = a + b*x
        sigma = ((b*xerr)**2 + yerr*2 + s**2)**0.5
        lglk = 2 * np.log(sigma).sum() + \
               (((y-model) / sigma)**2).sum() + \
               np.log(x.size) * (2*np.pi)**0.5 / 2
        return -lglk
    def lnprior(theta):
        """
        Prior. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_s
    def lnprob(theta, x, y, xerr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, y, xerr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
    start = np.array(start)
    ndim = start.size
    pos = np.random.normal(start, starting_width*start, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x1,x2,x1err,x2err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array(
        [sampler.chain[:,nburn:,i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().' \
                  ' Must be "full" or list of ints.'
            print(msg)
            exit()
    return



def mcmc_regression_2x(x1, x2, x3, x1err=None, x2err=None, x3err=None, start=(1.,1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=5000, nwalkers=100, nburn=0, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.
    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter
    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default True)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.
    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.
    """

    # just in case
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)

    if x1err is None:
        x1err = np.zeros(x1.size)
    if x2err is None:
        x2err = np.zeros(x1.size)
    if x3err is None:
        x3err = np.zeros(x1.size)

    def lnlike(theta, x, w, y, xerr, werr, yerr):
        """Likelihood"""
        a, b, c, s = theta
        model = a + b*x + c*w
        sigma = ((b*xerr)**2 + (c*werr)**2 + yerr*2 + s**2)**0.5
        lglk = 2 * np.log(sigma).sum() + \
               (((y-model) / sigma)**2).sum() + \
               np.log(x.size) * (2*np.pi)**0.5 / 2
        return -lglk
    def lnprior(theta):
        """
        Prior. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, c, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        lnp_c = np.log(stats.t.pdf(c, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_c + lnp_s
    def lnprob(theta, x, w, y, xerr, werr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, w, y, xerr, werr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
        x3, x3err = to_log(x3, x3err)
    start = np.array(start)
    ndim = start.size
    pos = np.random.normal(start, starting_width*start, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x1,x2,x3,x1err,x2err,x3err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array(
        [sampler.chain[:,nburn:,i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().' \
                  ' Must be "full" or list of ints.'
            print(msg)
            exit()
    return



def mcmc_regression_3x(x1, x2, x3, x4, x1err=None, x2err=None, x3err=None, x4err=None, start=(1.,1.,1.,1.,0.5),
         starting_width=0.01, logify=True,
         nsteps=5000, nwalkers=100, nburn=0, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.
    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter
    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default True)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.
    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.
    """

    # just in case
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)

    if x1err is None:
        x1err = np.zeros(x1.size)
    if x2err is None:
        x2err = np.zeros(x1.size)
    if x3err is None:
        x3err = np.zeros(x1.size)
    if x4err is None:
        x4err = np.zeros(x1.size)

    def lnlike(theta, x, w, z, y, xerr, werr, zerr, yerr):
        """Likelihood"""
        a, b, c, d, s = theta
        model = a + b*x + c*w +d*z
        sigma = ((b*xerr)**2 + (c*werr)**2 + (d*zerr)**2 + yerr*2 + s**2)**0.5
        lglk = 2 * np.log(sigma).sum() + \
               (((y-model) / sigma)**2).sum() + \
               np.log(x.size) * (2*np.pi)**0.5 / 2
        return -lglk
    def lnprior(theta):
        """
        Prior. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, c, d, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        lnp_c = np.log(stats.t.pdf(c, 1))
        lnp_d = np.log(stats.t.pdf(d, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_c + lnp_d + lnp_s
    def lnprob(theta, x, w, z, y, xerr, werr, zerr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, w, z, y, xerr, werr, zerr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
        x3, x3err = to_log(x3, x3err)
        x4, x4err = to_log(x4, x4err)
    start = np.array(start)
    ndim = start.size
    pos = np.random.normal(start, starting_width*start, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x1,x2,x3,x4,x1err,x2err,x3err,x4err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array(
        [sampler.chain[:,nburn:,i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().' \
                  ' Must be "full" or list of ints.'
            print(msg)
            exit()
    return


def mcmc_regression_4x(x1, x2, x3, x4, x5, x1err=None, x2err=None, x3err=None, x4err=None,  x5err=None, start=(1.,1.,1.,1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=5000, nwalkers=100, nburn=0, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.
    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter
    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default False)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.
    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.
    """

    # just in case
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    x5 = np.array(x5)

    if x1err is None:
        x1err = np.zeros(x1.size)
    if x2err is None:
        x2err = np.zeros(x1.size)
    if x3err is None:
        x3err = np.zeros(x1.size)
    if x4err is None:
        x4err = np.zeros(x1.size)
    if x5err is None:
        x5err = np.zeros(x1.size)

    def lnlike(theta, x, w, z, v, y, xerr, werr, zerr, verr,yerr):
        """Likelihood"""
        a, b, c, d, e, s = theta
        model = a + b*x + c*w + d*z + e*v
        sigma = ((b*xerr)**2 + (c*werr)**2 + (d*zerr)**2  + (e*verr)**2 + yerr*2 + s**2)**0.5
        lglk = 2 * np.log(sigma).sum() + \
               (((y-model) / sigma)**2).sum() + \
               np.log(x.size) * (2*np.pi)**0.5 / 2
        return -lglk
    def lnprior(theta):
        """
        Prior. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, c, d, e, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        lnp_c = np.log(stats.t.pdf(c, 1))
        lnp_d = np.log(stats.t.pdf(d, 1))
        lnp_e = np.log(stats.t.pdf(e, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_c + lnp_d + lnp_e + lnp_s
    def lnprob(theta, x, w, z, v, y, xerr, werr, zerr, verr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, w, z, v, y, xerr, werr, zerr, verr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
        x3, x3err = to_log(x3, x3err)
        x4, x4err = to_log(x4, x4err)
        x5, x5err = to_log(x5, x5err)
    start = np.array(start)
    ndim = start.size
    pos = np.random.normal(start, starting_width*start, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x1,x2,x3,x4,x5,x1err,x2err,x3err,x4err,x5err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array(
        [sampler.chain[:,nburn:,i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().' \
                  ' Must be "full" or list of ints.'
            print(msg)
            exit()
    return

def mcmc_regression_5x(x1, x2, x3, x4, x5, x6, x1err=None, x2err=None, x3err=None, x4err=None,  x5err=None, x6err=None, start=(1.,1.,1.,1.,1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=5000, nwalkers=100, nburn=0, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.
    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter
    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default False)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.
    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.
    """

    # just in case
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    x5 = np.array(x5)
    x6 = np.array(x6)

    if x1err is None:
        x1err = np.zeros(x1.size)
    if x2err is None:
        x2err = np.zeros(x1.size)
    if x3err is None:
        x3err = np.zeros(x1.size)
    if x4err is None:
        x4err = np.zeros(x1.size)
    if x5err is None:
        x5err = np.zeros(x1.size)
    if x6err is None:
        x6err = np.zeros(x1.size)

    def lnlike(theta, x, w, z, v, u, y, xerr, werr, zerr, verr, uerr, yerr):
        """Likelihood"""
        a, b, c, d, e, f, s = theta
        model = a + b*x + c*w + d*z + e*v + f*u
        sigma = ((b*xerr)**2 + (c*werr)**2 + (d*zerr)**2  + (e*verr)**2 + (f*uerr)**2 + yerr*2 + s**2)**0.5
        lglk = 2 * np.log(sigma).sum() + \
               (((y-model) / sigma)**2).sum() + \
               np.log(x.size) * (2*np.pi)**0.5 / 2
        return -lglk
    def lnprior(theta):
        """
        Prior. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, c, d, e, f, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        lnp_c = np.log(stats.t.pdf(c, 1))
        lnp_d = np.log(stats.t.pdf(d, 1))
        lnp_e = np.log(stats.t.pdf(e, 1))
        lnp_f = np.log(stats.t.pdf(f, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_c + lnp_d + lnp_e + lnp_f + lnp_s
    def lnprob(theta, x, w, z, v, u, y, xerr, werr, zerr, verr, uerr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, w, z, v, u, y, xerr, werr, zerr, verr, uerr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
        x3, x3err = to_log(x3, x3err)
        x4, x4err = to_log(x4, x4err)
        x5, x5err = to_log(x5, x5err)
        x6, x6err = to_log(x6, x6err)
    start = np.array(start)
    ndim = start.size
    pos = np.random.normal(start, starting_width*start, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x1,x2,x3,x4,x5,x6,x1err,x2err,x3err,x4err,x5err,x6err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array(
        [sampler.chain[:,nburn:,i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().' \
                  ' Must be "full" or list of ints.'
            print(msg)
            exit()
    return


'''
#test
# Reproducible results!

np.random.seed(123)

# Choose the "true" parameters.
m_true = 0.53
b_true = -0.85
c_true = -0.2
d_true = 0.17
f_true = 0.18

# Generate some synthetic data from the model.
N = 1000
x = np.sort(0.4+0.13*np.random.rand(N))
xerr = 0.0#0.005+0.001*np.random.rand(N)
x += xerr * np.random.randn(N)
w = np.sort(0.65+0.43*np.random.rand(N))
werr = 0.05+0.07*np.random.rand(N)
w += werr * np.random.randn(N)
z = np.sort(0.44+0.48*np.random.rand(N))
zerr = 0.07+0.09*np.random.rand(N)
z += zerr * np.random.randn(N)
yerr = 0.016+0.016*np.random.rand(N)
y = m_true*x+w*c_true+z*d_true+b_true
y += f_true*np.random.randn(N)#np.random.normal(0, f_true, N)
y += yerr * np.random.randn(N)
#y+= np.random.normal(0., np.random.normal(0.016, 0.016, N))
results=mcmc_regression_3x(x, w, z, y, x1err=xerr, x2err=werr, x3err=zerr,  x4err=yerr, start=(1.,1.,1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=500, nwalkers=100, nburn=100, output=(50,16,84))

print results


x = np.array([x,w,z]).T
x = sm.add_constant(x)
results = sm.WLS(endog=y,exog=x,weights=1/(yerr**2)).fit()
guess = results.params
print results.summary()
print results.conf_int()[1]


'''

np.random.seed(123)

# Choose the "true" parameters.
m_true = 0.6
b_true = -0.7
c_true = 0.3
d_true = 0.4
f_true = 0.2

# Generate some synthetic data from the model.
N = 1000
x = (1+0.13*np.random.rand(N))
xerr = 0.02+0.01*np.random.rand(N)
x += np.random.normal(0, xerr)#xerr*np.random.randn(N)
w = (1.5+0.43*np.random.rand(N))
werr = 0.02+0.01*np.random.rand(N)
w += np.random.normal(0, werr)#werr * np.random.randn(N)
z = (1.5+0.48*np.random.rand(N))
zerr = 0.02+0.02*np.random.rand(N)
z += np.random.normal(0, zerr)#zerr * np.random.randn(N)
yerr = 0.01+0.01*np.random.rand(N)
y = b_true+m_true*x+w*c_true#+z*d_true
y += np.random.normal(0, f_true, N)
y += np.random.normal(0, yerr)#yerr * np.random.randn(N)


#y+= np.random.normal(0., np.random.normal(0.016, 0.016, N))
#results=mcmc_regression_3x(x, w, z, y, x1err=xerr, x2err=werr, x3err=zerr,  x4err=yerr, start=(1.,1.,1.,1.,0.5),starting_width=0.01, logify=False,nsteps=500, nwalkers=100, nburn=100, output=(50,16,84))

results=mcmc_regression_2x(x, w, y, x1err=xerr, x2err=werr, x3err=yerr, start=(1.,1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=500, nwalkers=100, nburn=100, output=(50,16,84))

print results

'''

x = np.array([x,w,z]).T
x = sm.add_constant(x)
results = sm.WLS(endog=y,exog=x,weights=1/(yerr**2)).fit()
guess = results.params
print results.summary()
print results.conf_int()[1]


# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.234

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true) * np.random.randn(N)
y += yerr * np.random.randn(N)

results=mcmc_regression_1x(x,y,  x2err=yerr, start=(1.,1.,0.5),
         starting_width=0.01, logify=False,
         nsteps=500, nwalkers=100, nburn=100, output=(50,16,84))

print results

x = np.array([x]).T
x = sm.add_constant(x)
results = sm.WLS(endog=y,exog=x,weights=1/(yerr**2)).fit()
guess = results.params
print results.summary()
'''
