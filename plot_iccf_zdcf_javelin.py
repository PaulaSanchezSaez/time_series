import numpy as np
import os
import astropy.io.fits as pf
from multiprocessing import Pool
from matplotlib import pyplot as plt
import scipy
from scipy import stats



################################################################################
#file parameters

field='COSMOS' #COSMOS, XMM_LSS, ELAIS_S1, ECDFS
interp = 5.

#filters used
driving_filter='Q'
responding_filter='Y'

lc_path='../lowz_lc/final_catalog_well_sampled_type1_lc_Aug2018/'+field+'/'

zdcf_path='../zdcf_stat_Aug2018/'
iccf_path='../iccf_stat_Aug2018/'
out_file_name='../iccf_stat_Aug2018/results_lowz_ws_type1_'+field+'_'+driving_filter+'_'+responding_filter+'_dt'+str(interp)+'.txt'

dat=np.loadtxt(out_file_name,dtype='str').transpose()
ra=dat[0].astype(np.float)
dec=dat[1].astype(np.float)
zspec=dat[2].astype(np.float)
centau=dat[10].astype(np.float)
centau_uperr=dat[12].astype(np.float)
centau_loerr=dat[11].astype(np.float)

out_file_name_jav='../javelin_stat_Aug2018/results_lowz_ws_type1_'+field+'_'+driving_filter+'_'+responding_filter+'.txt'

dat=np.loadtxt(out_file_name_jav,dtype='str').transpose()
ra_jav=dat[0].astype(np.float)
dec_jav=dat[1].astype(np.float)
zspec_jav=dat[2].astype(np.float)
centau_jav=dat[4].astype(np.float)
centau_uperr_jav=dat[6].astype(np.float)
centau_loerr_jav=dat[5].astype(np.float)

significance_path='../significance_stat/'

for i in xrange(len(ra)):
    ras="%.6f" % ra[i]
    decs="%.6f" % dec[i]

    print "plotting ra=%s dec=%s" % (ras,decs)

    if zspec[i]>0:

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

        zdcf_cc=zdcf_path+'CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf'
        iccf_cc=iccf_path+'org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt5.0.txt'
        iccf_centdist=iccf_path+'centdist_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'_dt5.0.txt'

        tau_zdcf, tau_zdcf_low, tau_zdcf_up, r_zdcf, r_zdcf_low, r_zdcf_up =  np.loadtxt(zdcf_cc, unpack = True, usecols = [0, 1, 2, 3, 4, 5])
        tau_iccf, r_iccf =  np.loadtxt(iccf_cc, unpack = True, usecols = [0, 1])
        centdist=np.loadtxt(iccf_centdist, unpack = True, usecols = [0])

        sigma_jav, tau_jav, lag_jav, width_jav, scale_jav=np.loadtxt("../javelin_stat_Aug2018/jav_chain_all_"+driving_filter+"_vs_"+responding_filter+"_"+str(ras)+"_"+str(decs)+".txt", unpack = True, usecols = [0, 1, 2, 3, 4])

        try: sig_lag,sig_val=np.loadtxt(significance_path+"lag_significance_"+driving_filter+"_vs_"+responding_filter+"_"+str(ras)+"_"+str(decs)+".txt", unpack = True, usecols = [0,1])
        except: sig_lag,sig_val= np.linspace(-500,1000,100) , np.zeros(100)

        fig = plt.figure()
        fig.set_size_inches(15,10)
        fig.subplots_adjust(hspace=0.2, wspace = 0.25)

        #Plot lightcurves
        ax1 = fig.add_subplot(2,1,1)
        ax1.errorbar(jd_driving/(zspec_driving+1.0), flux_driving-np.mean(flux_driving)+0.05, yerr = errflux_driving, marker = '.', linestyle = ':', color = 'b',ecolor='b',label=driving_filter)
        ax1.errorbar(jd_responding/(zspec_driving+1.0), flux_responding-np.mean(flux_responding), yerr = errflux_responding, marker = '.', linestyle = ':', color = 'r',ecolor='r',label=responding_filter)
        ax1.set_title('ra=%s dec=%s z=%f' % (ras,decs,zspec_driving))
        ax1.legend()
        ax1.set_ylabel('Normaliced Flux')
        ax1.set_xlabel('MJD')

        print "lcs plotted"

        #Plot CCF Information
        #xmin, xmax = lag_range[0],lag_range[1]
        ax2 = fig.add_subplot(2, 3, 4)
        ax2.set_ylabel('CCF r')
        ax2.text(0.15, 0.85, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
        #ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(-1.0, 1.0)
        ax2.plot(tau_iccf, r_iccf, 'bo-',label='ICCF')
        ax2.errorbar(tau_zdcf, r_zdcf, xerr=[tau_zdcf_low, tau_zdcf_up], yerr = [r_zdcf_low, r_zdcf_up], marker = '*', linestyle = ':', color = 'r',ecolor='r',label='ZDCF')
        ax2.set_xlabel('Centroid Lag: %5.1f (+%5.1f -%5.1f) days'%(centau[i], centau_uperr[i], centau_loerr[i]), fontsize = 15)
        plt.plot(np.linspace(-5000,5000,8),np.ones(8)*0.5,'k--')
        plt.plot(np.zeros(8),np.linspace(-1,1,8),'k--')
        ax2.set_xlim(np.min(tau_zdcf),np.max(tau_zdcf))
        ax2.legend()
        print "r vs tau plotted"


        ax3 = fig.add_subplot(2, 3, 5)
        color = 'tab:blue'
        #ax3.set_xlim(xmin, xmax)
        #ax3.axes.get_yaxis().set_ticks([])
        n, bins, etc = ax3.hist(centdist, bins = 50, color = 'b')
        ax3.text(0.05, 0.95, 'CCCD ICCF', horizontalalignment = 'left', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 14)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_xlim(-400,900)
        print "iccf centdist plotted"

        ax3_2 = ax3.twinx()
        color = 'tab:red'
        ax3_2.plot(sig_lag,sig_val,'r--')
        ax3_2.tick_params(axis='y', labelcolor=color)

        ax4 = fig.add_subplot(2, 3, 6)
        color = 'tab:green'
        n, bins, etc = ax4.hist(lag_jav, bins = 50, color = 'g')
        ax4.text(0.05, 0.95, 'CCCD Javelin', horizontalalignment = 'left', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 14)
        ax4.set_xlim(-400,900)
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.set_xlabel('Jav. Lag: %5.1f (+%5.1f -%5.1f) days'%(centau_jav[i], centau_uperr_jav[i], centau_loerr_jav[i]), fontsize = 15)

        ax4_2 = ax4.twinx()
        color = 'tab:red'
        ax4_2.plot(sig_lag,sig_val,'r--')
        ax4_2.tick_params(axis='y', labelcolor=color)

        print "javelin centdist plotted"


        plt.savefig('../cc_plots_Aug2018/results_zdcf_iccf_javelin_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.png', format = 'png', orientation = 'landscape', bbox_inches = 'tight')
        plt.close('all')

    elif (zspec_jav[i]>0):

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

        zdcf_cc=zdcf_path+'CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf'
        iccf_cc=iccf_path+'org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt'
        iccf_centdist=iccf_path+'centdist_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt'

        tau_zdcf, tau_zdcf_low, tau_zdcf_up, r_zdcf, r_zdcf_low, r_zdcf_up =  np.loadtxt(zdcf_cc, unpack = True, usecols = [0, 1, 2, 3, 4, 5])
        #tau_iccf, r_iccf =  np.loadtxt(iccf_cc, unpack = True, usecols = [0, 1])
        #centdist=np.loadtxt(iccf_centdist, unpack = True, usecols = [0])

        sigma_jav, tau_jav, lag_jav, width_jav, scale_jav=np.loadtxt("../javelin_stat_Aug2018/jav_chain_all_"+driving_filter+"_vs_"+responding_filter+"_"+str(ras)+"_"+str(decs)+".txt", unpack = True, usecols = [0, 1, 2, 3, 4])

        try: sig_lag,sig_val=np.loadtxt(significance_path+"lag_significance_"+driving_filter+"_vs_"+responding_filter+"_"+str(ras)+"_"+str(decs)+".txt", unpack = True, usecols = [0,1])
        except: sig_lag,sig_val= np.linspace(-500,1000,100) , np.zeros(100)

        fig = plt.figure()
        fig.set_size_inches(13,10)
        fig.subplots_adjust(hspace=0.2, wspace = 0.1)

        #Plot lightcurves
        ax1 = fig.add_subplot(2,1,1)
        ax1.errorbar(jd_driving/(zspec_driving+1.0), flux_driving-np.mean(flux_driving)+0.05, yerr = errflux_driving, marker = '.', linestyle = ':', color = 'b',ecolor='b',label=driving_filter)
        ax1.errorbar(jd_responding/(zspec_driving+1.0), flux_responding-np.mean(flux_responding), yerr = errflux_responding, marker = '.', linestyle = ':', color = 'r',ecolor='r',label=responding_filter)
        ax1.set_title('ra=%s dec=%s z=%f' % (ras,decs,zspec_driving))
        ax1.legend()
        ax1.set_ylabel('Normaliced Flux')
        ax1.set_xlabel('MJD')

        print "lcs plotted"

        #Plot CCF Information
        #xmin, xmax = lag_range[0],lag_range[1]
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.set_ylabel('r')
        ax2.text(0.15, 0.85, 'CC', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
        #ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(-1.0, 1.0)
        #ax2.plot(tau_iccf, r_iccf, 'bo-',label='ICCF')
        ax2.errorbar(tau_zdcf, r_zdcf, xerr=[tau_zdcf_low, tau_zdcf_up], yerr = [r_zdcf_low, r_zdcf_up], marker = '*', linestyle = ':', color = 'r',ecolor='r',label='ZDCF')
        #ax2.set_xlabel('Centroid Lag: %5.1f (+%5.1f -%5.1f) days'%(centau[i], centau_uperr[i], centau_loerr[i]), fontsize = 15)
        plt.plot(np.linspace(-5000,5000,8),np.ones(8)*0.5,'k--')
        plt.plot(np.zeros(8),np.linspace(-1,1,8),'k--')
        ax2.set_xlim(np.min(tau_zdcf),np.max(tau_zdcf))
        ax2.legend()
        print "r vs tau plotted"



        ax3 = fig.add_subplot(2, 2, 4)
        color = 'tab:green'
        n, bins, etc = ax3.hist(lag_jav, bins = 50, color = 'g')
        ax3.text(0.15, 0.85, 'CCCD Javelin', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 14)
        ax3.set_xlim(-400,900)
        ax3.set_xlabel('Jav. Lag: %5.1f (+%5.1f -%5.1f) days'%(centau_jav[i], centau_uperr_jav[i], centau_loerr_jav[i]), fontsize = 15)
        ax1.tick_params(axis='y', labelcolor=color)

        ax3_2 = ax3.twinx()
        color = 'tab:red'
        ax3_2.plot(sig_lag,sig_val,'r--')
        ax3_2.tick_params(axis='y', labelcolor=color)
        print "javelin centdist plotted"

        plt.savefig('../cc_plots_Aug2018/results_zdcf_iccf_javelin_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.png', format = 'png', orientation = 'landscape', bbox_inches = 'tight')
        plt.close('all')
