import numpy as np
import carmcmc as cm
import matplotlib.pyplot as plt
from os import environ
import cPickle
from scipy.optimize import curve_fit
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model

def carma_order(date, mag, mag_err,maxp,aic_file):
    #function to calculate the order p and q of the CARMA(p,q) process
    # maxp: Maximum value allowed for p, maximun value for q is by default p-1

    date=date-date[0]

    model = cm.CarmaModel(date, mag, mag_err)
    MAP, pqlist, AIC_list = model.choose_order(maxp, njobs=1)

    # convert lists to a numpy arrays, easier to manipulate
    #the results of the AIC test are stored for future references.
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    np.savetxt(aic_file,np.transpose([pmodels,qmodels,AICc]),header='p  q  AICc')


    pparam = model.p
    qparam = model.q

    return (pparam,qparam)


#functions used to fit the power-law to the PSD
def broken_power_low_log(x, amplitude, x_break, gamma_1, gamma_2):
        """One dimensional broken power law model function"""

        gamma = np.where(x < x_break, gamma_1, gamma_2)
        xx = x / x_break
        return (np.log10(amplitude) + (gamma * np.log10(xx)))

def broken_power_low(x, amplitude, x_break, gamma_1, gamma_2):
        """One dimensional broken power law model function"""

        gamma = np.where(x < x_break, gamma_1, gamma_2)
        xx = x / x_break
        return (amplitude*(xx**gamma))




def fit_broken_pl(freq, low_psd,mid_psd,high_psd):
    xdata=np.log10(freq)
    ydata=np.log10(mid_psd)
    yerr=0.5*(np.log10(high_psd)-np.log10(low_psd))

    print xdata
    print ydata
    print yerr

    plt.errorbar(xdata,ydata,yerr=yerr,fmt='b*')
    plt.show()

    #popt, pcov = curve_fit(broken_power_low_log, xdata, ydata,sigma=yerr,method='lm')
    #popt, pcov = curve_fit(broken_power_low_log, xdata, ydata,method='lm')
    popt, pcov = curve_fit(broken_power_low, freq, mid_psd,method='lm')
    perr = np.sqrt(np.diag(pcov))

    return (popt,perr)



"""
class BendingPL(Fittable1DModel):
    A = Parameter()
    v_bend = Parameter()
    a_low = Parameter()
    a_high = Parameter()


    @staticmethod
    def evaluate(x, A,v_bend,a_low,a_high):
        '''
        Bending power law function - returns power at each value of v,
        where v is an array (e.g. of frequencies)

        inputs:
            v (array)       - input values
            A (float)       - normalisation
            v_bend (float)  - bending frequency
            a_low ((float)  - low frequency index
            a_high float)   - high frequency index
        output:
            out (array)     - output powers
        '''
        numer = x**-a_low
        denom = 1. + (x/v_bend)**(a_high-a_low)
        out = A * (numer/denom)
        return out

    @staticmethod
    def fit_deriv(x, A,v_bend,a_low,a_high):
        d_A = (x**-a_low)/(1. + (x/v_bend)**(a_high-a_low))

        d_v_bend = -1.0*(A*(x**-a_low)/(1. + (x/v_bend)**(a_high-a_low))**2)*((((x)**(a_high-a_low))*(a_high-a_low))/(v_bend**(a_high-a_low-1)))

        d_a_low = (-1.0*a_low*A*(x**(-a_low-1.)))/(1. + (x/v_bend)**(a_high-a_low)) + (a_low*A*(x**-a_low)*((x/v_bend)**(a_high-a_low-1)))/(1. + (x/v_bend)**(a_high-a_low))**2

        d_a_high = -1.0*a_high*((x/v_bend)**(a_high-a_low-1.))*A*(x**-a_low)/(1. + (x/v_bend)**(a_high-a_low))**2

        return [d_A,d_v_bend,d_a_low,d_a_high]

"""


@custom_model

def BendingPL(x, A=1.0,v_bend=0.0025,a_low=1,a_high=3):
    '''
    Bending power law function - returns power at each value of v,
    where v is an array (e.g. of frequencies)

    inputs:
        v (array)       - input values
        A (float)       - normalisation
        v_bend (float)  - bending frequency
        a_low ((float)  - low frequency index
        a_high float)   - high frequency index
    output:
        out (array)     - output powers
    '''
    numer = x**-a_low
    denom = 1. + (x/v_bend)**(a_high-a_low)
    out = A * (numer/denom)
    return out


def fit_BendingPL(freq,psd):

    bpl_init = BendingPL()
    fit = LevMarLSQFitter()
    bpl = fit(bpl_init, freq, psd)


    return (bpl.A.value,bpl.v_bend.value,bpl.a_low.value,bpl.a_high.value,bpl(freq))



def run_CARMA(time, y, ysig,maxp,nsamples,aic_file,carma_sample_file,psd_file,psd_plot,fit_quality_plot,pl_plot,do_mags=True):
    #to calculate the order p and q of the CARMA(p,q) process, then run CARMA for values of p and q already calculated


    #function to calculate the order p and q of the CARMA(p,q) process
    # maxp: Maximum value allowed for p, maximun value for q is by default p-1

    time=time-time[0]

    model = cm.CarmaModel(time, y, ysig)
    MAP, pqlist, AIC_list = model.choose_order(maxp, njobs=1)

    # convert lists to a numpy arrays, easier to manipulate
    #the results of the AIC test are stored for future references.
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    np.savetxt(aic_file,np.transpose([pmodels,qmodels,AICc]),header='p  q  AICc')


    p = model.p
    q = model.q


    #running the sampler
    carma_model = cm.CarmaModel(time, y, ysig, p=p, q=q)
    carma_sample = carma_model.run_mcmc(nsamples)
    carma_sample.add_mle(MAP)

    #getting the PSD
    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False, color='SkyBlue', nsamples=5000)

    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))

    #saving the psd
    np.savetxt(psd_file,np.transpose([frequencies,psd_low, psd_hi, psd_mid,psd_mle]),header='frequencies  psd_low  psd_hi  psd_mid psd_mle')

    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    dt = time[1:] - time[0:-1]
    noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)
    mean_noise_level=noise_level
    median_noise_level=2.0 * np.median(dt) * np.median(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.loglog(frequencies, np.ones(frequencies.size) * median_noise_level, color='green', lw=2)

    ax.set_ylim(bottom=noise_level / 100.0)

    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency [1 / day]')
    if do_mags:
        ax.set_ylabel('Power Spectral Density [mag$^2$ day]')
    else:
        ax.set_ylabel('Power Spectral Density [flux$^2$ day]')
    #plt.title(title)
    plt.savefig(psd_plot)
    plt.close('all')


    print 'Assessing the fit quality...'
    fig = carma_sample.assess_fit(doShow=False)
    ax_again = fig.add_subplot(2, 2, 1)
    #ax_again.set_title(title)
    if do_mags:
        ylims = ax_again.get_ylim()
        ax_again.set_ylim(ylims[1], ylims[0])
        ax_again.set_ylabel('magnitude')
    else:
        ax_again.set_ylabel('ln Flux')
    plt.savefig(fit_quality_plot)

    pfile = open(carma_sample_file, 'wb')
    cPickle.dump(carma_sample, pfile)
    pfile.close()

    params = {param: carma_sample.get_samples(param) for param in carma_sample.parameters}
    params['p'] = model.p
    params['q'] = model.q


    print "fitting bending power-law"
    nf=np.where(psd_mid>=median_noise_level)

    psdfreq=frequencies[nf]
    psd_low=psd_low[nf]
    psd_hi=psd_hi[nf]
    psd_mid=psd_mid[nf]

    A,v_bend,a_low,a_high,blpfit=fit_BendingPL(psdfreq,psd_mid)

    pl_init = models.BrokenPowerLaw1D(amplitude=2, x_break=0.002, alpha_1=1, alpha_2=2)
    fit = LevMarLSQFitter()
    pl = fit(pl_init, psdfreq, psd_mid)

    amplitude=pl.amplitude.value
    x_break=pl.x_break.value
    alpha_1=pl.alpha_1.value
    alpha_2=pl.alpha_2.value

    print amplitude,x_break,alpha_1,alpha_2

    print "BendingPL fit parameters = ",A,v_bend,a_low,a_high
    print "BrokenPL fit parameters = ",amplitude,x_break,alpha_1,alpha_2


    plt.clf()
    plt.subplot(111)
    plt.loglog(psdfreq, psd_mid, color='green')
    plt.fill_between(psdfreq, psd_low, psd_hi, facecolor='green', alpha=0.3)
    plt.plot(psdfreq,blpfit,'r--',lw=2)
    plt.plot(psdfreq,pl(psdfreq),'k--',lw=2)
    plt.savefig(pl_plot)
    plt.close('all')

    return (params, mean_noise_level, median_noise_level, A,v_bend,a_low,a_high,amplitude,x_break,alpha_1,alpha_2)
