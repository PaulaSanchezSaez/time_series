#this code contain all the functions to calculate the variability features for a light curve
import numpy as np
import astropy.io.fits as pf
import os
#import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.integrate import quad
import emcee
import corner
import scipy.stats as st
from scipy.stats import chi2
import FATS
import carmcmc as cm

##########################################
#to get P, exvar y exvar_err

def var_parameters(jd,mag,err):
#function to calculate the probability of a light curve to be variable, and the excess variance

    #nepochs, maxmag, minmag, mean, variance, skew, kurt = st.describe(mag)

    mean=np.mean(mag)
    nepochs=float(len(jd))

    chi= np.sum( (mag - mean)**2. / err**2. )
    q_chi=chi2.cdf(chi,(nepochs-1))


    a=(mag-mean)**2
    ex_var=(np.sum(a-err**2)/((nepochs*(mean**2))))
    sd=np.sqrt((1./(nepochs-1))*np.sum(((a-err**2)-ex_var*(mean**2))**2))
    ex_verr=sd/((mean**2)*np.sqrt(nepochs))
    #ex_var=(np.sum(a)/((nepochs-1)*(mean**2)))-(np.sum(err**2)/(nepochs*(mean**2)))

    #ex_verr=np.sqrt(((np.sqrt(2./nepochs)*np.sum(err**2)/(mean*nepochs))**2)+((np.sqrt(np.sum(err**2)/nepochs**2)*2*np.sqrt(ex_var)/mean)**2))

    #ex_verr=np.sqrt((np.sqrt(2.0/nepochs)*np.mean(err*2)/mean**2)**2+(np.sqrt(np.mean(err*2)/nepochs)*(2.0*np.sqrt(ex_var)/mean))**2)

    #print q_chi,ex_var,ex_verr

    return [q_chi,ex_var,ex_verr]



#######################################
#determine single SF using emcee

def SFarray(jd,mag,err):#calculate an array with (m(t)-m(t+tau)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt
    sfarray=[]
    tauarray=[]
    errarray=[]
    for i, item in enumerate(mag):
        for j in range(i+1,len(mag)):
            dm=mag[i]-mag[j]
            sigma=err[i]**2+err[j]**2
            dt=(jd[j]-jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray=np.array(sfarray)
    tauarray=np.array(tauarray)
    errarray=np.array(errarray)
    return (tauarray,sfarray,errarray)


def Vmod(dt,A,gamma): #model
    return ( A*((dt/365.0)**gamma) )
    #return ( A*(dt**gamma) )


def Veff2(dt,sigma,A,gamma): #model plus the error
    return ( (Vmod(dt,A,gamma))**2 + sigma )

def like_one(theta,dt,dmag,sigma): #likelihood for one value of dmag

    gamma, A = theta
    aux=(1/np.sqrt(2*np.pi*Veff2(dt,sigma,A,gamma)))*np.exp(-1.0*(dmag**2)/(2.0*Veff2(dt,sigma,A,gamma)))

    return aux

def lnlike(theta, dtarray, dmagarray, sigmaarray): # we define the likelihood following the same function used by Schmidt et al. 2010
    gamma, A = theta

    '''
    aux=0.0

    for i in xrange(len(dtarray)):
    aux+=np.log(like_one(theta,dtarray[i],dmagarray[i],sigmaarray[i]))
    '''

    aux=np.sum(np.log(like_one(theta,dtarray,dmagarray,sigmaarray)))

    return aux



def lnprior(theta): # we define the prior following the same functions implemented by Schmidt et al. 2010

    gamma, A = theta


    if 0.0 < gamma and 0.0 < A < 2.0 :
        return ( np.log(1.0/A) + np.log(1.0/(1.0+(gamma**2.0))) )

    return -np.inf
    #return -(10**32)


def lnprob(theta, dtarray, dmagarray, sigmaarray): # the product of the prior and the likelihood in a logaritmic format

    lp = lnprior(theta)

    if not np.isfinite(lp):
    #if (lp==-(10**32)):
        return -np.inf
        #return -(10**32)
    return lp +lnlike(theta, dtarray, dmagarray, sigmaarray)


def fitSF_mcmc(jd,mag,errmag,ndim,nwalkers,nit,nthr): #function that fits the values of A and gamma using mcmc with the package emcee.
#It recives the array with dt in days, dmag and the errors, besides the number of dimensions of the parameters, the number of walkers and the number of iterations

    #we calculate the arrays of dm, dt and sigma
    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)


    #plt.plot(dtarray,dmagarray,'bo')
    #plt.show()

    ndt=np.where((dtarray<=365) & (dtarray>=10))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]

    #plt.plot(dtarray,dmagarray,'bo')
    #plt.show()

    #definition of the optimal initial position of the walkers

    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to start the burn in fase

    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [0.1,0.1], args=(x, y, yerr))

    #run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthr, args=(dtarray, dmagarray, sigmaarray))

    pos, prob, state = sampler.run_mcmc(p0,100) #from pos we have a best gess of the initial walkers
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit,rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    #ac=sampler.acceptance_fraction

    #plt.hist(ac)
    #plt.show()

    #print samples

    A_fin=samples[:,1]
    gamma_fin=samples[:,0]

    #print gamma_fin
    #print A_fin

    #A_hist,A_bins=np.histogram(A_fin,200)
    #a_max=np.amax(A_hist)
    #na=np.where(A_hist==a_max)
    #a_mode=A_bins[na]

    #print "a_mode", a_mode

    #g_hist,g_bins=np.histogram(gamma_fin,200)
    #g_max=np.amax(g_hist)
    #ng=np.where(g_hist==g_max)
    #g_mode=g_bins[ng]

    #print "g_mode", g_mode

    #plt.plot(A_fin,gamma_fin,'.')
    #plt.show()

    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
    #fig.savefig("line-triangle.png")
    #plt.show()

    #plt.hist(A_fin,100)
    #plt.xlabel('A')
    #plt.show()


    #plt.hist(gamma_fin,100)
    #plt.xlabel('gamma')
    #plt.show()

    A_mcmc=(np.percentile(A_fin, 50),np.percentile(A_fin, 50)-np.percentile(A_fin, 15.865),np.percentile(A_fin, 84.135)-np.percentile(A_fin, 50))
    g_mcmc=(np.percentile(gamma_fin, 50),np.percentile(gamma_fin, 50)-np.percentile(gamma_fin, 15.865),np.percentile(gamma_fin, 84.135)-np.percentile(gamma_fin, 50))

    #print A_mcmc
    #print g_mcmc
    sampler.reset()
    return (A_mcmc, g_mcmc)



#######################################
#determine ensamble SF using emcee


def SFlist(jd,mag,err):#calculate an array with (m(t)-m(t+tau)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt
    sfarray=[]
    tauarray=[]
    errarray=[]
    for i, item in enumerate(mag):
        for j in range(i+1,len(mag)):
            dm=mag[i]-mag[j]
            sigma=err[i]**2+err[j]**2
            dt=(jd[j]-jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)

    return (tauarray,sfarray,errarray)

def SF_array_ensambled(list_objects,filt,magname,errname,nbin):

    dtlist=[]
    dmaglist=[]
    derrlist=[]



    #the dt, dmag and derr are determined for all the objects

    for i in xrange(len(list_objects)):

        arch=pf.open('../'+filt+'/'+list_objects[i])
        dat=arch[1].data
        head=arch[0].header
        z=head['ZSPEC']
        jd=dat['JD']
        mag=dat[magname]
        err=dat[errname]

        dta,dmaga,derra=SFlist(jd/(1+z),mag,err)

        dtlist+=dta
        dmaglist+=dmaga
        derrlist+=derra

        arch.close()


    dt=np.array(dtlist)
    derr=(np.array(derrlist))
    dmag=(np.array(dmaglist))

    return (dt,dmag,derr)




def fitSF_mcmc_ensambled(list_objects,filt,magname,errname,nbin,ndim,nwalkers,nit,nthr): #function that fits the values of A and gamma using mcmc with the package emcee.
#It recives the array with dt in days, dmag and the errors, besides the number of dimensions of the parameters, the number of walkers and the number of iterations

    #we calculate the arrays of dm, dt and sigma

    dtarray, dmagarray, sigmaarray = SF_array_ensambled(list_objects,filt,magname,errname,nbin)


    '''
    #test to explore the parameter space
    a_array=np.linspace(0,1,500)
    g_array=np.linspace(0,2,500)

    test_post=np.zeros((500,500))

    for i in xrange(len(g_array)):
        for j in xrange(len(a_array)):

            test_post[i,j]=np.exp(lnprob([g_array[i],a_array[j]], dtarray, dmagarray, sigmaarray))

    plt.contourf(a_array,g_array,test_post,100)
    plt.show()

    nn=np.where(test_post==np.amax(test_post))
    gg=g_array[nn[0]]
    aa=a_array[nn[1]]
    return (aa,gg)
    '''

    #definition of the optimal initial position of the walkers

    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to start the burn in fase

    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [0.1,0.1], args=(x, y, yerr))

    #run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=nthr, args=(dtarray, dmagarray, sigmaarray))

    pos, prob, state = sampler.run_mcmc(p0,50) #from pos we have a best gess of the initial walkers
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit,rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    ac=sampler.acceptance_fraction

    #plt.hist(ac)
    #plt.show()

    #print samples

    A_fin=samples[:,1]
    gamma_fin=samples[:,0]

    print gamma_fin
    print A_fin

    #plt.plot(A_fin,gamma_fin,'.')
    #plt.show()

    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
    #fig.savefig("line-triangle.png")

    #plt.hist(A_fin,100)
    #plt.show()

    #plt.hist(gamma_fin,100)
    #plt.show()

    A_mcmc=(np.percentile(A_fin, 50),np.percentile(A_fin, 50)-np.percentile(A_fin, 15.865),np.percentile(A_fin, 84.135)-np.percentile(A_fin, 50))
    g_mcmc=(np.percentile(gamma_fin, 50),np.percentile(gamma_fin, 50)-np.percentile(gamma_fin, 15.865),np.percentile(gamma_fin, 84.135)-np.percentile(gamma_fin, 50))
    sampler.reset()
    return (A_mcmc, g_mcmc)




##################################################
#run FATS

def run_fats(jd, mag, err):
#function tu run fats and return the features in an array
    #list with the features to be calculated
    feature_list = [
        #'Mean',
        #'Std',
        #'Meanvariance',
        #'MedianBRP',
        #'Rcs',
        #'PeriodLS',
        #'Period_fit',
        #'Color',
        'Autocor_length',
        #'SlottedA_length',
        #'StetsonK',
        #'StetsonK_AC',
        #'Eta_e',
        #'Amplitude',
        #'PercentAmplitude',
        #'Con',
        #'LinearTrend',
        #'Beyond1Std',
        #'FluxPercentileRatioMid20',
        #'FluxPercentileRatioMid35',
        #'FluxPercentileRatioMid50',
        #'FluxPercentileRatioMid65',
        #'FluxPercentileRatioMid80',
        #'PercentDifferenceFluxPercentile',
        #'Q31',
        'CAR_sigma',
        'CAR_mean',
        'CAR_tau',
    ]


    data_array = np.array([mag, jd, err])
    data_ids = ['magnitude', 'time', 'error']
    feat_space = FATS.FeatureSpace(featureList=feature_list, Data=data_ids)
    feat_vals = feat_space.calculateFeature(data_array)
    f_results = feat_vals.result(method='array')
    f_features = feat_vals.result(method='features')

    return (f_results,f_features)




def all_single_features(jd,mag,err,ndim,nwalkers,nit,nthr):

    print "calculating var parameters"
    P,exvar , exvar_err = var_parameters(jd,mag,err)
    print "calculating SF"
    A_mcmc , g_mcmc = fitSF_mcmc(jd,mag,err,ndim,nwalkers,nit,nthr)
    print "running FATS"
    fats_features = run_fats(jd, mag, err)

    return [P, exvar, exvar_err , A_mcmc, g_mcmc, fats_features[0]]





def run_carmcmc(date, mag, mag_err):

    # Maximum order of the autoregressive polynomial
    MAX_ORDER_AR = 6
    # Number of samples from the posterior to generate
    N_SAMPLES = 20000

    model = cm.CarmaModel(date, mag, mag_err)
    model.choose_order(MAX_ORDER_AR, njobs=-1)
    sample = model.run_mcmc(N_SAMPLES)

    psd_low, psd_hi, psd_mid, frequencies = sample.plot_power_spectrum(percentile=95.0, nsamples=5000)
    dt = t[1:] - t[:-1]
    noise_level_mean = 2.0 * np.mean(dt) * np.mean(yerr ** 2)
    noise_level_median = 2.0 * np.median(dt) * np.median(yerr ** 2)

    print "psd"
    print psd_mid
    print "frec"
    print frequencies

    params = {param: sample.get_samples(param) for param in sample.parameters}
    params['p'] = model.p
    params['q'] = model.q

    return params


def bincalc(nbin,bmin,bmax): #calculate the bin range

    logbmin=np.log10(bmin)
    logbmax=np.log10(bmax)

    logbins=np.arange(logbmin,logbmax,nbin)

    bins=10**logbins

    #bins=np.linspace(bmin,bmax,60)
    return (bins)

def SF_K16(jd,mag,errmag,nbin,bmin,bmax,phot_noise=True):
    #method to calculate the SF using the definition given by Kozlowski 2016a. Eq 20.
    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)

    bins=bincalc(nbin,bmin,bmax)


    iqr_list=[]
    tau_list=[]
    numobj_list=[]

    for i in range(0,len(bins)-1):
        n=np.where((dtarray>=bins[i]) & (dtarray<bins[i+1]))
        nobjbin=len(n[0])
        if nobjbin>=6:
            numobj_list.append(nobjbin)
            dmag_bin=dmagarray[n]

            #the IQR is calculated
            q75, q25 = np.percentile(dmag_bin, [75 ,25])
            iqr = q75 - q25
            iqr_list.append(iqr)

            #central tau for the bin
            tau_list.append((bins[i]+bins[i+1])*0.5)

    iqr_list=np.array(iqr_list)
    tau_list=np.array(tau_list)
    numobj_list=np.array(numobj_list)

    '''
    print iqr_list, tau_list

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(tau_list,0.741*iqr_list,'ro')
    plt.xlabel('tau')
    plt.ylabel('SF10')
    plt.show()


    plt.plot(tau_list,numobj_list,'bo')
    plt.xlabel('tau')
    plt.ylabel('num_bin')
    plt.show()
    '''

    nn=np.where(numobj_list>=6)
    iqr_list=iqr_list[nn]
    tau_list=tau_list[nn]


    if phot_noise:
        nnoise=np.where((dtarray<=2.0))
        num_noise=len(nnoise[0])
        #print "num dt<2 = ", num_noise
        noise_dmag= dmagarray[nnoise]
        noise_dtau= dtarray[nnoise]

        #plt.plot(noise_dtau,noise_dmag,'g*')
        #plt.show()

        q75n, q25n = np.percentile(noise_dmag, [75 ,25])
        iqrn = q75n - q25n
        #print "iqrnoise = ",iqrn

        SF2=np.sqrt(0.549*(iqr_list**2-iqrn**2))
        nsf=np.where(SF2>-100)
        SF2=SF2[nsf]
        tau_list=tau_list[nsf]

        #plt.plot(tau_list,(SF2),'ro')
        #plt.show()

        return (tau_list/365.0,(SF2))

    else:

        SF2=0.741*(iqr_list)
        nsf=np.where(SF2>-100)
        SF2=SF2[nsf]
        tau_list=tau_list[nsf]
        return (tau_list/365.0,(SF2))




def gen_rand_lc(jd,mag,err): #generate a random lc from a given lc

    new_mag=[]

    for i in xrange(len(jd)):
        nm=np.random.normal(mag[i], err[i])
        new_mag.append(nm)

    new_mag=np.array(new_mag)

    return(new_mag)


def SFfin_K16(jd,mag,errmag,nbin,bmin,bmax,numit):#calculate the final SF distribution for a given light curve, using numit random light curves defined from the original lc.
    """
    jd,mag, errmag are the light curve
    min is the minimun tau considered
    max is the lenght of the light curve or the maximun tau considered
    dt is the resolution of the light curve
    numit is the number of iteration where the random light curves are generated
    """


    list_SF=[]#list to save the SF values

    for j in xrange(numit): #for every iteration the new light curve and Sf are generated
        new_mag=gen_rand_lc(jd,mag,errmag)
        tau2,sf=SF_K16(jd,new_mag,errmag,nbin,bmin,bmax)
        list_SF.append(sf)
        #print "iteration %d" % (j)

    array_SF=np.array(list_SF)

    #print array_SF
    #print np.shape(array_SF)

    SF=[]#array to save the final value of SF
    err_up=[]#array to save the upper error
    err_low=[]#array to save the lower error


    for i in xrange(len(tau2)):
        sf=array_SF[:,i]
        med_sf=np.median(sf)
        eup=np.percentile(sf,84.135)
        elo=np.percentile(sf,15.865)
        SF.append(med_sf)
        err_up.append(eup)
        err_low.append(elo)

        '''
        print med_sf, elo,eup
        plt.hist(sf,50)
        plt.plot(np.ones(4)*med_sf,np.array([0,5,10,20]),'r-')
        plt.plot(np.ones(4)*eup,np.array([0,5,10,20]),'r-')
        plt.plot(np.ones(4)*elo,np.array([0,5,10,20]),'r-')
        plt.show()
        '''

    SF=np.array(SF)
    err_up=np.array(err_up)
    err_low=np.array(err_low)

    sffin=np.sqrt(SF)
    errlow=0.5*(SF-err_low)/np.sqrt(SF)
    errup=0.5*(err_up-SF)/np.sqrt(SF)

    nsf=np.where(sffin>-100)
    sffin=sffin[nsf]
    errlow=errlow[nsf]
    errup=errup[nsf]
    tau2=tau2[nsf]

    #plt.errorbar(tau2,sffin,yerr=[errlow,errup],fmt='k.',ecolor='k')
    #plt.show()

    return(tau2,sffin,errlow,errup)

def fitSF(tau,sf): #fit the model A*tau^gamma to the SF, the fit only consider the bins with more than 6 pairs.

    y=np.log10(sf)
    x=np.log10(tau)
    x=x[np.where((tau<=1) & (tau>0.01))]
    y=y[np.where((tau<=1) & (tau>0.01))]
    coefficients = np.polyfit(x, y, 1)

    A=10**(coefficients[1])
    gamma=coefficients[0]

    return(A,gamma)


def SFSchmidt10(jd,mag,errmag,nbin,bmin,bmax): #calculate the SF for a nbin number of bins, returns (tau,SF)
    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)

    bins=bincalc(nbin,bmin,bmax)
    print len(bins)


    sf_list=[]
    tau_list=[]
    numobj_list=[]

    for i in range(0,len(bins)-1):
        n=np.where((dtarray>=bins[i]) & (dtarray<bins[i+1]))
        nobjbin=len(n[0])
        if nobjbin>=6:
            dmag1=np.abs(dmagarray[n])
            derr1=np.sqrt(sigmaarray[n])
            sf=(np.sqrt(np.pi/2.0)*dmag1-derr1)
            sff=np.mean(sf)
            sf_list.append(sff)
            numobj_list.append(nobjbin)
            #central tau for the bin
            tau_list.append((bins[i]+bins[i+1])*0.5)




    SF=np.array(sf_list)
    nob=np.array(numobj_list)
    tau=np.array(tau_list)
    nn=np.where(nob>6)
    tau=tau[nn]
    SF=SF[nn]

    return (tau/365.,SF) #nob is the number of pairs considered per bin to calculate the SF
