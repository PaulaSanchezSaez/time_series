import numpy as np
import lc_simulation as lc
import os
import astropy.io.fits as pf
from multiprocessing import Pool
import sys
import argparse
from matplotlib import pyplot as plt
import scipy
import PYCCF as myccf
from scipy import stats
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model,Rmap_Model,Pmap_Model

################################################################################
#simulation parameters:
nsimulations=100
zspec=0.0
real_lag=100
dmag=0.5
time_range=2000
dtime=3
mag=20
errmag1=0.02
errmag2=0.04
tau=400
SFinf=0.3
sampling=True
timesamp1_name='agn_150.176559_1.759390_Y.fits'
timesamp2_name='agn_150.176559_1.759390_Ks.fits'
save_file='stat/agn_150.176559_1.759390_sampling'+'_zpec'+str(zspec)+'_lag'+str(real_lag)+'_err1'+str(errmag1)+'_err2'+str(errmag2)+'_tau'+str(tau)+'_SFinf'+str(SFinf)+'.txt'

#parameters that are not important for the simulation, they are used for consistency
driving_filter='Y'
responding_filter='Ks'

################################################################################
#Javelin cross correlation parameters:

cent_lowlimit=-400
cent_uplimit=900
perclim = 84.1344746

################################################################################
#ICCF cross correlation parameters:

lag_range = [-400, 900]  #Time lag range to consider in the CCF (days). Must be small enough that there is some overlap between light curves at that shift (i.e., if the light curves span 80 days, these values must be less than 80 days)
interp = 5. #Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
nsim = 10000  #Number of Monte Carlo iterations for calculation of uncertainties
mcmode = 0  #Do both FR/RSS sampling (1 = RSS only, 2 = FR only)
sigmode = 0.5  #Choose the threshold for considering a measurement "significant". sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed". See code for different sigmodes.

################################################################################

def do_iccf(source_name,source_ID,zspec_driving,jd_driving,flux_driving,errflux_driving,jd_responding,flux_responding,errflux_responding):
    #Calculate lag with python CCF program

    print "########## computing ICCF for source %d  ##########" % (source_ID)

    tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = myccf.peakcent(jd_driving/(zspec_driving+1.0), flux_driving, jd_responding/(zspec_driving+1.0), flux_responding, lag_range[0], lag_range[1], interp)
    tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = myccf.xcor_mc(jd_driving/(zspec_driving+1.0), flux_driving, abs(errflux_driving), jd_responding/(zspec_driving+1.0), flux_responding, abs(errflux_responding), lag_range[0], lag_range[1], interp, nsim = nsim, mcmode=mcmode, sigmode = sigmode)

    lag = ccf_pack[1]
    r = ccf_pack[0]

    ###Calculate the best peak and centroid and their uncertainties using the median of the
    ##distributions.
    centau = stats.scoreatpercentile(tlags_centroid, 50)
    centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim))-centau
    centau_loerr = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim)))
    print 'Centroid, error: %10.3f  (+%10.3f -%10.3f)'%(centau, centau_uperr, centau_loerr)
    print "centroid org:", tlag_centroid
    peaktau = stats.scoreatpercentile(tlags_peak, 50)
    peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim))-peaktau
    peaktau_loerr = peaktau-(stats.scoreatpercentile(tlags_peak, (100.-perclim)))
    print 'Peak, errors: %10.3f  (+%10.3f -%10.3f)'%(peaktau, peaktau_uperr, peaktau_loerr)
    print "peak org:", tlag_peak

    #Write results out to a file in case we want them later.
    centfile = open('iccf_stat/centdist_iccf_dt'+str(interp)+'_'+source_name+'.txt', 'w')
    peakfile = open('iccf_stat/peakdist_iccf_dt'+str(interp)+'_'+source_name+'.txt', 'w')
    ccf_file = open('iccf_stat/org_iccf_dt'+str(interp)+'_'+source_name+'.txt', 'w')
    for m in xrange(0, np.size(tlags_centroid)):
        centfile.write('%5.5f    \n'%(tlags_centroid[m]))
    centfile.close()
    for m in xrange(0, np.size(tlags_peak)):
        peakfile.write('%5.5f    \n'%(tlags_peak[m]))
    peakfile.close()
    for m in xrange(0, np.size(lag)):
        ccf_file.write('%5.5f    %5.5f  \n'%(lag[m], r[m]))
    ccf_file.close()

    return (source_ID,zspec_driving,nsuccess_peak,tlag_peak,peaktau,peaktau_loerr,peaktau_uperr,nsuccess_centroid,tlag_centroid,centau,centau_loerr,centau_uperr)

################################################################################
def do_javelin(source_name,source_ID,zspec_driving,jd_driving,flux_driving,errflux_driving,jd_responding,flux_responding,errflux_responding):
    #Calculate lag with python Javelin program

    print "########## computing Javelin for source %d  ##########" % (source_ID)

    #converting lcs into a format accepted by the javelin method
    aa=str(np.random.randint(100))
    lcd_name='temp/driving_lc_'+aa+'_'+str(source_ID)+'.txt'
    np.savetxt(lcd_name,np.transpose([jd_driving/(1.0+zspec_driving),flux_driving,errflux_driving]))

    lcr_name='temp/responding_lc_'+aa+'_'+str(source_ID)+'.txt'
    np.savetxt(lcr_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))

    #running Javelin
    cont=get_data([lcd_name],names=[driving_filter])
    cmod=Cont_Model(cont)
    cmod.do_mcmc(nwalkers=100, nburn=50, nchain=100, fchain="javelin_stat/chain_cont_"+source_name+".txt")

    bothdata=get_data([lcd_name,lcr_name],names=[driving_filter,responding_filter])
    mod_2band=Pmap_Model(bothdata)#Rmap_Model(bothdata)
    mod_2band.do_mcmc(nwalkers=100, nburn=50, nchain=100,conthpd=cmod.hpd,laglimit=[[cent_lowlimit,cent_uplimit]],fchain="javelin_stat/jav_chain_all_"+source_name+".txt")

    sigma, tau, lag, width, scale=np.loadtxt("javelin_stat/jav_chain_all_"+source_name+".txt", unpack = True, usecols = [0, 1, 2, 3, 4])

    centau_median = np.median(lag)
    centau_uperr = (stats.scoreatpercentile(lag, perclim))-centau_median
    centau_loerr = centau_median-(stats.scoreatpercentile(lag, (100.-perclim)))
    len_chain=len(lag[np.where(lag>-2000000000000)])

    return (source_ID,zspec_driving,len_chain,centau_median,centau_loerr,centau_uperr)

################################################################################

def get_sampling(filename):
    a=pf.open(filename)
    dat=a[1].data
    jd=dat['JD']
    mag=dat['MAG_2']
    err=dat['MAGERR_2']

    return((jd-jd[0]),np.mean(mag),np.mean(err))

################################################################################
#function that simulates lcs and compute lags with iccf and javelin

def sim_lc_do_cc(source_ID,source_name,zspec,real_lag,dmag,time_range,dtime,mag,errmag1,errmag2,tau,SFinf,sampling,timesamp1,timesamp2):

    #simulating driving (1) and responding (2) lcs:
    t_drw1,y_obs1,ysig_obs1,t_drw2,y_obs2,ysig_obs2=lc.gen_DRW_long_DRM(source_ID,int(real_lag*(1+zspec)),dmag,time_range,dtime,mag,errmag1,errmag2,tau,SFinf,sampling,timesamp1,timesamp2)

    #running ICCF

    source_ID,zspec_driving,nsuccess_peak,tlag_peak,peaktau,peaktau_loerr,peaktau_uperr,nsuccess_centroid,tlag_centroid,centau,centau_loerr,centau_uperr=do_iccf(source_name,source_ID,zspec,t_drw1,y_obs1,ysig_obs1,t_drw2,y_obs2,ysig_obs2)

    source_ID,zspec_driving,jav_len_chain,jav_centau_median,jav_centau_loerr,jav_centau_uperr=do_javelin(source_name,source_ID,zspec_driving,t_drw1,y_obs1,ysig_obs1,t_drw2,y_obs2,ysig_obs2)

    return(source_ID,zspec_driving,real_lag,nsuccess_peak,tlag_peak,peaktau,peaktau_loerr,peaktau_uperr,nsuccess_centroid,tlag_centroid,centau,centau_loerr,centau_uperr,jav_len_chain,jav_centau_median,jav_centau_loerr,jav_centau_uperr)

################################################################################
#we run the simulations, using the parameters defined at the begining of the code


tobs_Y,mag_Y,err_Y=get_sampling(timesamp1_name)
tobs_Ks,mag_Ks,err_Ks=get_sampling(timesamp2_name)

result_list=[] #where the results of all the simulations are saved

for i in xrange(nsimulations):
    print "simulating source ", i
    source_name='agn'+str(i)+'_zpec'+str(zspec)+'_lag'+str(real_lag)+'_err1'+str(errmag1)+'_err2'+str(errmag2)+'_tau'+str(tau)+'_SFinf'+str(SFinf)
    result=sim_lc_do_cc(i,source_name,zspec,real_lag,dmag,time_range,dtime,mag,errmag1,errmag2,tau,SFinf,sampling,tobs_Y,tobs_Ks)

    result_list.append(result)

results=np.array(result_list)

head='ID  zspec  real_lag  iccf_nsuccess_peak  iccf_peak_org iccf_peak_median  iccf_peak_lowerr  iccf_peak_uperr  iccf_nsuccess_centroid  iccf_centroid_org  iccf_centroid_median  iccf_cent_lowerr  iccf_cent_uperr  jav_len_chain  jav_lag_median  jav_lag_lowerr  jav_lag_uperr'
np.savetxt(save_file,results,header=head)
print "File %s writen" % (save_file)
