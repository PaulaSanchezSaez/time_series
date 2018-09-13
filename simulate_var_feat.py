import numpy as np
#from matplotlib import pyplot as plt
import astropy.io.fits as pf
from astroML.time_series import lomb_scargle, generate_damped_RW
from astroML.time_series import ACF_scargle, ACF_EK
import carmcmc as cm
import corner
from matplotlib.ticker import NullFormatter
import lc_simulation as lc
from all_features import *
from multiprocessing import Pool
import time


def run_SF(t, y_obs, ysig,nwalkers,nit,phot_noise):
    t = t - t[0]

    #mcmc SF
    try:
        A_mcmc,gamma_mcmc=fitSF_mcmc(t,y_obs,ysig,2,nwalkers,nit,1)
    except:
        A_mcmc,gamma_mcmc=(bp.array([-99,-99,-99]),np.array([-99,-99,-99]))
    #K16 SF
    try:
        tauk16,SFk16=SF_K16(t,y_obs,ysig,0.1,10,2000,phot_noise=phot_noise)
        Ak16,gammak16=fitSF(tauk16,SFk16)
    except:
        Ak16,gammak16=(-99,-99)

    try:
        taus10,SFs10=SFSchmidt10(t,y_obs,ysig,0.1,10,2000)
        As10,gammas10=fitSF(taus10,SFs10)
    except:
        As10,gammas10=(-99,-99)

    #plt.loglog(tauk16,SFk16,'b*')
    #plt.loglog(taus10,SFs10,'ro')
    #plt.loglog(tauk16,Ak16*(tauk16**gammak16),'b-')
    #plt.loglog(taus10,As10*(taus10**gammas10),'r-')
    #plt.show()

    return (A_mcmc,gamma_mcmc,Ak16,gammak16,As10,gammas10)


def run_car1(t, y_obs, ysig,nsamples):
    t = t - t[0]
    model = cm.CarmaModel(t, y_obs, ysig, p=1, q=0)
    sample = model.run_mcmc(nsamples)
    log_omega=sample.get_samples('log_omega')
    tau=np.exp(-1.0*log_omega)
    sigma=sample.get_samples('sigma')
    #ax = plt.subplot(111)
    #psd_low, psd_hi, psd_mid, frequencies = sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False, color='SkyBlue', nsamples=5000)
    #plt.show()

    #compute the mode of the distributions of tau and sigma
    nt, bins=np.histogram(tau,30)
    binst = (bins[1:] + bins[0:-1])*0.5
    elemt = np.argmax(nt)
    mode_tau=binst[elemt]


    ns, bins=np.histogram(sigma,30)
    binss = (bins[1:] + bins[0:-1])*0.5
    elems = np.argmax(ns)
    mode_sigma=binss[elems]

    tau_mc=(mode_tau,np.percentile(tau, 50),np.percentile(tau, 50)-np.percentile(tau, 15.865),np.percentile(tau, 84.135)-np.percentile(tau, 50))
    sigma_mc=(mode_sigma,np.percentile(sigma, 50),np.percentile(sigma, 50)-np.percentile(sigma, 15.865),np.percentile(sigma, 84.135)-np.percentile(sigma, 50))

    return (np.array(tau_mc),np.array(sigma_mc))




def simulate_drw_SF(seed,time_range,dtime,mag,errmag,tau,SFinf,nwalkers_SF,nit_SF,phot_noise,nsamples_DRW,do_SF,do_DRW,sampling,timesamp):
    #code that simulate a light curve and measures its DRW parameters and the Structure Fuction.

    #we simulate the light curve
    t,y,ysig,y_obs,ysig_obs=lc.gen_DRW_long(seed,time_range,dtime,mag,errmag,tau,SFinf,sampling,timesamp)

    print do_SF, do_DRW

    #variability parameters
    print "calculating var parameters"
    pvar,exvar,err_exvar = var_parameters(t,y,ysig)
    pvar_obs,exvar_obs,err_exvar_obs = var_parameters(t,y_obs,ysig_obs)


    #SF parameters
    if do_SF:
        print "calculating SF paramenters "
        try:
            A_mcmc,gamma_mcmc,Ak16,gammak16,As10,gammas10=run_SF(t,y,ysig,nwalkers_SF,nit_SF,phot_noise)

        except:
            print "################fail in SF procedure...################"
            A_mcmc,gamma_mcmc,Ak16,gammak16,As10,gammas10=(np.array([-99,-99,-99]),np.array([-99,-99,-99]),-99,-99,-99,-99)

        try:
            A_mcmc_obs,gamma_mcmc_obs,Ak16_obs,gammak16_obs,As10_obs,gammas10_obs=run_SF(t,y_obs,ysig_obs,nwalkers_SF,nit_SF,phot_noise)
        except:
            print "################fail in SF procedure...################"
            A_mcmc_obs,gamma_mcmc_obs,Ak16_obs,gammak16_obs,As10_obs,gammas10_obs=(np.array([-99,-99,-99]),np.array([-99,-99,-99]),-99,-99,-99,-99)
    #DRW parameters
    if do_DRW:
        print "calculating DRW parameters"
        try:
            tau_mc,sigma_mc=run_car1(t,y,ysig,nsamples_DRW)

        except:
            print "################fail in DRW procedure...################"
            tau_mc,sigma_mc=(np.array([-99,-99,-99]),np.array([-99,-99,-99]))

        try:
            tau_mc_obs,sigma_mc_obs=run_car1(t,y_obs,ysig_obs,nsamples_DRW)
        except:
            print "################fail in DRW procedure...################"
            tau_mc_obs,sigma_mc_obs=(np.array([-99,-99,-99]),np.array([-99,-99,-99]))




    num_epochs=len(t)
    time_range_out=t[-1]-t[0]

    print "seed ", seed
    print "lc properties", num_epochs, time_range_out
    print "var parameters ",  pvar,exvar,err_exvar
    if do_SF: print "SFinf_in = ", str(SFinf), "  run ", "A = ",Ak16,As10, A_mcmc, "gamma = ",gammak16,gammas10, gamma_mcmc
    if do_SF: print "SFinf_in = ", str(SFinf), "  run ", "A_obs = ",Ak16_obs,As10_obs, A_mcmc_obs, "gamma = ",gammak16_obs,gammas10_obs, gamma_mcmc_obs

    if do_DRW: print "tau_in = ", str(tau), "tau = ", tau_mc, "sig = ", sigma_mc
    if do_DRW: print "tau_in = ", str(tau), "tau_obs = ", tau_mc_obs, "sig_obs = ", sigma_mc_obs



    if do_SF and do_DRW: return (seed,num_epochs,time_range_out,tau,SFinf,pvar,exvar,err_exvar,gammak16,gammas10,gamma_mcmc[0],gamma_mcmc[1],gamma_mcmc[2],Ak16,As10,A_mcmc[0],A_mcmc[1],A_mcmc[2],tau_mc[0],tau_mc[1],tau_mc[2],tau_mc[3],sigma_mc[0],sigma_mc[1],sigma_mc[2],sigma_mc[3],pvar_obs,exvar_obs,err_exvar_obs,gammak16_obs,gammas10_obs,gamma_mcmc_obs[0],gamma_mcmc_obs[1],gamma_mcmc_obs[2],Ak16_obs,As10_obs,A_mcmc_obs[0],A_mcmc_obs[1],A_mcmc_obs[2],tau_mc_obs[0],tau_mc_obs[1],tau_mc_obs[2],tau_mc_obs[3],sigma_mc_obs[0],sigma_mc_obs[1],sigma_mc_obs[2],sigma_mc_obs[3])
    elif (do_SF==True) and (do_DRW==False): return (seed,num_epochs,time_range_out,tau,SFinf,pvar,exvar,err_exvar,gammak16,gammas10,gamma_mcmc[0],gamma_mcmc[1],gamma_mcmc[2],Ak16,As10,A_mcmc[0],A_mcmc[1],A_mcmc[2],pvar_obs,exvar_obs,err_exvar_obs,gammak16_obs,gammas10_obs,gamma_mcmc_obs[0],gamma_mcmc_obs[1],gamma_mcmc_obs[2],Ak16_obs,As10_obs,A_mcmc_obs[0],A_mcmc_obs[1],A_mcmc_obs[2])
    elif (do_DRW==True) and (do_SF==False): return (seed,num_epochs,time_range_out,tau,SFinf,pvar,exvar,err_exvar,tau_mc[0],tau_mc[1],tau_mc[2],tau_mc[3],sigma_mc[0],sigma_mc[1],sigma_mc[2],sigma_mc[3],pvar_obs,exvar_obs,err_exvar_obs,tau_mc_obs[0],tau_mc_obs[1],tau_mc_obs[2],tau_mc_obs[3],sigma_mc_obs[0],sigma_mc_obs[1],sigma_mc_obs[2],sigma_mc_obs[3])
    else: return (seed,num_epochs,time_range_out,tau,SFinf,pvar,exvar,err_exvar,pvar_obs,exvar_obs,err_exvar_obs)




def multi_run_wrapper(args):
#function necessary for the use of pool.map with different arguments
   return simulate_drw_SF(*args)


def run_sim(nsim,ncores,time_range,dtime,mag,errmag,tau,SFinf,nwalkers_SF,nit_SF,phot_noise,nsamples_DRW,do_SF,do_DRW,sampling,timesamp,save_file):

    arg_list=[]
    #the array with the arguments is generated, this is necessary to use pool.map
    for i in range(0,nsim):
        arg_list.append((i,time_range,dtime,mag,errmag,tau,SFinf,nwalkers_SF,nit_SF,phot_noise,nsamples_DRW,do_SF,do_DRW,sampling,timesamp))

    pool = Pool(processes=ncores)
    results = pool.map(multi_run_wrapper,arg_list)
    pool.close()
    pool.join()

    if do_SF and do_DRW:  head='seed  num_epochs  time_range_out  tau  SFinf  pvar  exvar  err_exvar  gammak16  gammas10  gamma_mcmc  gamma_loerr gamma_uperr  Ak16 As10  A_mcmc  A_loerr  A_uperr  tau_mode  tau_mc  tau_loerr  tau_uperr sigma_mode  sigma_mc  sigma_loerr  sigma_uperr  pvar_obs  exvar_obs  err_exvar_obs  gammak16_obs  gammas10_obs  gamma_mcmc_obs  gamma_loerr_obs gamma_uperr_obs  Ak16_obs As10_obs  A_mcmc_obs  A_loerr_obs  A_uperr_obs  tau_mode_obs  tau_mc_obs tau_loerr_obs  tau_uperr_obs sigma_mode_obs  sigma_mc_obs  sigma_loerr_obs sigma_uperr_obs'
    elif (do_SF==True) and (do_DRW==False): head='seed  num_epochs  time_range_out  tau  SFinf  pvar  exvar  err_exvar  gammak16  gammas10  gamma_mcmc  gamma_loerr gamma_uperr  Ak16 As10  A_mcmc  A_loerr  A_uperr  pvar_obs  exvar_obs  err_exvar_obs  gammak16_obs  gammas10_obs  gamma_mcmc_obs  gamma_loerr_obs gamma_uperr_obs  Ak16_obs As10_obs  A_mcmc_obs  A_loerr_obs  A_uperr_obs'
    elif (do_SF==False) and (do_DRW==True) : head='seed  num_epochs  time_range_out  tau  SFinf  pvar  exvar  err_exvar  tau_mode  tau_mc  tau_loerr  tau_uperr sigma_mode  sigma_mc  sigma_loerr  sigma_uperr    pvar_obs  exvar_obs  err_exvar_obs  tau_mode_obs  tau_mc_obs tau_loerr_obs  tau_uperr_obs sigma_mode_obs  sigma_mc_obs  sigma_loerr_obs sigma_uperr_obs  '
    else: head='seed  num_epochs  time_range_out  tau  SFinf  pvar  exvar  err_exvar   pvar_obs  exvar_obs  err_exvar_obs'

    np.savetxt(save_file,results,header=head)
    return (results)




def get_sampling(filename):
    a=pf.open(filename)
    dat=a[1].data
    jd=dat['JD']
    mag=dat['Q']
    err=dat['errQ']

    return((jd-jd[0]),np.mean(mag),np.mean(err))


samp_file_short='bin3_onechip_30.595546_-1.961014_XMM_LSS.fits'
samp_file_med='bin3_morechip_151.393520_2.171140_COSMOS.fits'
samp_file_long='bin3_onechip_32.912634_-4.435199_XMM_LSS.fits'

tobs_short,mag_short,err_short=get_sampling(samp_file_short)
tobs_med,mag_med,err_med=get_sampling(samp_file_med)
tobs_long,mag_long,err_long=get_sampling(samp_file_long)
tobs_super_long=np.concatenate((tobs_long,tobs_long+tobs_long[-1]+200))





#short sampling
results0=run_sim(1000,20,1700,3,19,0.02,100,0.2,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau100_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.2,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau300_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.2,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau600_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.2,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau1000_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,100,0.4,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau100_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.4,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau300_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.4,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau600_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.4,50,200,True,50000,True,True,True,tobs_short,'var_feat_shortQUESTsampling_tau1000_SFinf04_err002_SF_DRW.txt')
del results0


#med sampling
results0=run_sim(1000,20,1700,3,19,0.02,100,0.2,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau100_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.2,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau300_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.2,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau600_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.2,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau1000_SFinf02_err002_SF_DRW.txt')
del results0


results0=run_sim(1000,20,1700,3,19,0.02,100,0.4,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau100_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.4,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau300_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.4,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau600_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.4,50,200,True,50000,True,True,True,tobs_med,'var_feat_medQUESTsampling_tau1000_SFinf04_err002_SF_DRW.txt')
del results0



#long sampling
results0=run_sim(1000,20,1700,3,19,0.02,100,0.2,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau100_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.2,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau300_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.2,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau600_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.2,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau1000_SFinf02_err002_SF_DRW.txt')
del results0


results0=run_sim(1000,20,1700,3,19,0.02,100,0.4,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau100_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.4,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau300_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.4,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau600_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.4,50,200,True,50000,True,True,True,tobs_long,'var_feat_longQUESTsampling_tau1000_SFinf04_err002_SF_DRW.txt')
del results0


#super_long sampling
results0=run_sim(1000,20,1700,3,19,0.02,100,0.2,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau100_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.2,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau300_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.2,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau600_SFinf02_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.2,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau1000_SFinf02_err002_SF_DRW.txt')
del results0


results0=run_sim(1000,20,1700,3,19,0.02,100,0.4,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau100_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,300,0.4,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau300_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,600,0.4,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau600_SFinf04_err002_SF_DRW.txt')
del results0

results0=run_sim(1000,20,1700,3,19,0.02,1000,0.4,50,200,True,50000,True,True,True,tobs_super_long,'var_feat_super_longQUESTsampling_tau1000_SFinf04_err002_SF_DRW.txt')
del results0
