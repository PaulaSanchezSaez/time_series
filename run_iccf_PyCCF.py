#code to run calculate iccf, using the code PyCCF (Cython version)

import numpy as np
import os
import astropy.io.fits as pf
from multiprocessing import Pool
import sys
import argparse
from matplotlib import pyplot as plt
import scipy
import PYCCF as myccf
from scipy import stats


################################################################################
##Set Interpolation settings, user-specified
################################################################################

lag_range = [-400, 900]  #Time lag range to consider in the CCF (days). Must be small enough that there is some overlap between light curves at that shift (i.e., if the light curves span 80 days, these values must be less than 80 days)
interp = 5. #Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
nsim = 10000  #Number of Monte Carlo iterations for calculation of uncertainties
mcmode = 0  #Do both FR/RSS sampling (1 = RSS only, 2 = FR only)
sigmode = 0.5  #Choose the threshold for considering a measurement "significant". sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed". See code for different sigmodes.
perclim = 84.1344746

################################################################################
#file parameters

field='COSMOS' #COSMOS, XMM_LSS, ELAIS_S1, ECDFS

#filters used
driving_filter='Q'
responding_filter='J'

#list of sources, in ra and dec
source_list='../stat/lowz_ws_type1_'+driving_filter+'_list_COSMOS.txt'

lc_path='../lowz_lc/final_catalog_well_sampled_type1_lc_Aug2018/'+field+'/'

out_file_name='../iccf_stat_Aug2018/results_lowz_ws_type1_'+field+'_'+driving_filter+'_'+responding_filter+'_dt'+str(interp)+'.txt'


n_cores=1

dat=np.loadtxt(source_list,dtype='str').transpose()
ra=dat[0].astype(np.float)
dec=dat[1].astype(np.float)
################################################################################

################################################################################
#we define a function to calculate the cross correlation per every source, and estimate the errors
def do_iccf(ra,dec,plotting_results):
    #ra and dec are converted to 6 decimals format
    ras="%.6f" % ra
    decs="%.6f" % dec

    print "########## computing iccf for source located at ra=%f and dec=%f  ##########" % (ra, dec)

    try:

        #the diver lc is loaded, we take into account the different formats used for opt and NIR data.
        if driving_filter!='Q':
            agn_driving=lc_path+driving_filter+'/agn_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.fits'
            arch_driving=pf.open(agn_driving)
            jd_0=55000
            head_driving=arch_driving[0].header
            datos_driving=arch_driving[1].data
            jd_driving=datos_driving['JD']-jd_0
            flux_driving=datos_driving['FLUX_2']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
            errflux_driving=datos_driving['FLUXERR_2']*1e27
            zspec_driving=head_driving['REDSHIFT']

        else:
            try:
                agn_driving=lc_path+driving_filter+'/bin3_onechip_'+str(ras)+'_'+str(decs)+'_'+field+'.fits'
                arch_driving=pf.open(agn_driving)
            except:
                agn_driving=lc_path+driving_filter+'/bin3_morechip_'+str(ras)+'_'+str(decs)+'_'+field+'.fits'
                arch_driving=pf.open(agn_driving)
            jd_0=2455000
            head_driving=arch_driving[0].header
            datos_driving=arch_driving[1].data
            jd_driving=datos_driving['JD']-jd_0
            flux_driving=datos_driving['fluxQ']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
            errflux_driving=datos_driving['errfluxQ']*1e27
            zspec_driving=head_driving['REDSHIFT']


        try:

            #reading the responding filter data
            if responding_filter!='Q':
                agn_responding=lc_path+responding_filter+'/agn_'+str(ras)+'_'+str(decs)+'_'+responding_filter+'.fits'
                arch_responding=pf.open(agn_responding)
                jd_0=55000
                head_responding=arch_responding[0].header
                datos_responding=arch_responding[1].data
                jd_responding=datos_responding['JD']-jd_0
                flux_responding=datos_responding['FLUX_2']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
                errflux_responding=datos_responding['FLUXERR_2']*1e27
                zspec_responding=head_responding['REDSHIFT']
            else:
                try:
                    agn_driving=lc_path+responding_filter+'/bin3_onechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                except:
                    agn_driving=lc_path+responding_filter+'/bin3_morechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                jd_0=2455000
                head_responding=arch_responding[0].header
                datos_responding=arch_responding[1].data
                jd_responding=datos_responding['JD']-jd_0
                flux_responding=datos_responding['fluxQ']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
                errflux_responding=datos_responding['errfluxQ']*1e27
                zspec_responding=head_responding['REDSHIFT']



            #Calculate lag with python CCF program
            tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = myccf.peakcent(jd_driving/(zspec_driving+1.0), flux_driving, jd_responding/(zspec_driving+1.0), flux_responding, lag_range[0], lag_range[1], interp)
            tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = myccf.xcor_mc(jd_driving/(zspec_driving+1.0), flux_driving, abs(errflux_driving), jd_responding/(zspec_driving+1.0), flux_responding, abs(errflux_responding), lag_range[0], lag_range[1], interp, nsim = nsim, mcmode=mcmode, sigmode = sigmode)

            lag = ccf_pack[1]
            r = ccf_pack[0]

            ###Calculate the best peak and centroid and their uncertainties using the median of the
            ##distributions.
            centau = stats.scoreatpercentile(tlags_centroid, 50)
            centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim))-centau
            centau_loerr = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim)))
            print 'Centroid, error: %10.3f  (+%10.3f -%10.3f)'%(centau, centau_loerr, centau_uperr)
            print "centroid org:", tlag_centroid
            peaktau = stats.scoreatpercentile(tlags_peak, 50)
            peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim))-centau
            peaktau_loerr = centau-(stats.scoreatpercentile(tlags_peak, (100.-perclim)))
            print 'Peak, errors: %10.3f  (+%10.3f -%10.3f)'%(peaktau, peaktau_uperr, peaktau_loerr)
            print "peak org:", tlag_peak

            #Write results out to a file in case we want them later.
            centfile = open('../iccf_stat_Aug2018/centdist_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt'+str(interp)+'.txt', 'w')
            peakfile = open('../iccf_stat_Aug2018/peakdist_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt'+str(interp)+'.txt', 'w')
            ccf_file = open('../iccf_stat_Aug2018/org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt'+str(interp)+'.txt', 'w')
            for m in xrange(0, np.size(tlags_centroid)):
                centfile.write('%5.5f    \n'%(tlags_centroid[m]))
            centfile.close()
            for m in xrange(0, np.size(tlags_peak)):
                peakfile.write('%5.5f    \n'%(tlags_peak[m]))
            peakfile.close()
            for m in xrange(0, np.size(lag)):
                ccf_file.write('%5.5f    %5.5f  \n'%(lag[m], r[m]))
            ccf_file.close()

            if plotting_results:
                #plotting the results.
                #Plot the Light curves, CCF, CCCD, and CCPD

                fig = plt.figure()
                fig.subplots_adjust(hspace=0.2, wspace = 0.1)

                #Plot lightcurves
                ax1 = fig.add_subplot(3, 1, 1)
                ax1.errorbar(jd_driving/(zspec_driving+1.0), flux_driving, yerr = errflux_driving, marker = '.', linestyle = ':', color = 'k')
                ax1_2 = fig.add_subplot(3, 1, 2, sharex = ax1)
                ax1_2.errorbar(jd_responding/(zspec_driving+1.0), flux_responding, yerr = errflux_responding, marker = '.', linestyle = ':', color = 'k')
                ax1.set_title('ra=%s dec=%s z=%f' % (ras,decs,zspec_driving))

                ax1.text(0.025, 0.825, driving_filter, fontsize = 15, transform = ax1.transAxes)
                ax1_2.text(0.025, 0.825, responding_filter, fontsize = 15, transform = ax1_2.transAxes)
                ax1.set_ylabel('LC 1 Flux')
                ax1_2.set_ylabel('LC 2 Flux')
                ax1_2.set_xlabel('MJD')

                #Plot CCF Information
                xmin, xmax = lag_range[0],lag_range[1]
                ax2 = fig.add_subplot(3, 3, 7)
                ax2.set_ylabel('CCF r')
                ax2.text(0.2, 0.85, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(-1.0, 1.0)
                ax2.plot(lag, r, color = 'k')

                ax3 = fig.add_subplot(3, 3, 8, sharex = ax2)
                ax3.set_xlim(xmin, xmax)
                ax3.axes.get_yaxis().set_ticks([])
                ax3.set_xlabel('Centroid Lag: %5.1f (+%5.1f -%5.1f) days'%(centau, centau_uperr, centau_loerr), fontsize = 15)
                ax3.text(0.25, 0.85, 'CCCD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 16)
                n, bins, etc = ax3.hist(tlags_centroid, bins = 50, color = 'b')

                ax4 = fig.add_subplot(3, 3, 9, sharex = ax2)
                ax4.set_ylabel('N')
                ax4.yaxis.tick_right()
                ax4.yaxis.set_label_position('right')
                #ax4.set_xlabel('Lag (days)')
                ax4.set_xlim(xmin, xmax)
                ax4.text(0.25, 0.85, 'CCPD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 16)
                ax4.hist(tlags_peak, bins = bins, color = 'b')

                plt.savefig('../iccf_plots/results_plot_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt'+str(interp)+'.png', format = 'png', orientation = 'landscape', bbox_inches = 'tight')
                plt.close('all')


            return (ra,dec,zspec_driving,nsuccess_peak,tlag_peak,peaktau,peaktau_loerr,peaktau_uperr,nsuccess_centroid,tlag_centroid,centau,centau_loerr,centau_uperr)

        except:
            print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO RESPONDING LC available  ##########" % (ra, dec)
            return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999)
    except:

        print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO DRIVING LC available  ##########" % (ra, dec)
        return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999)


################################################################################
#we run the cross correlation using multiprocessing

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return do_iccf(*args)

def cc_parallel(ncores,ra_array,dec_array,save_file):
    #creating the list used by multi_run_wrapper
    #arg_list=[]
    result_list=[]
    for i in xrange(len(ra_array)):
        #arg_list.append((ra_array[i],dec_array[i],True))
        result=do_iccf(ra_array[i],dec_array[i],True)
        result_list.append(result)
    #pool = Pool(processes=ncores)
    #results = pool.map(multi_run_wrapper,arg_list)
    #pool.close()
    #pool.join()
    results=np.array(result_list)
    head='ra dec zspec nsuccess_peak  peak_org peak_median  peak_lowerr  peak_uperr  nsuccess_centroid  centroid_org  centroid_median  cent_lowerr  cent_uperr'
    np.savetxt(save_file,results,header=head)
    print "File %s writen" % (save_file)
    return (results)

################################################################################

cc_parallel(n_cores,ra,dec,out_file_name)
#results=do_iccf(150.443665,2.049100,True)
#results=do_iccf(149.44458,2.11992,True)
#print results
