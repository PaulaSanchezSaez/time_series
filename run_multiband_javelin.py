from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model,Rmap_Model,Pmap_Model
import numpy as np
import os
import astropy.io.fits as pf
from multiprocessing import Pool
import sys
import argparse
from matplotlib import pyplot as plt
import scipy
from scipy import stats


################################################################################
#cross correlation parameters:

cent_lowlimit=-100
cent_uplimit=600
perclim = 84.1344746

################################################################################
#file parameters

field='COSMOS' #COSMOS, XMM_LSS, ELAIS_S1, ECDFS

#filters used
driving_filter='Y'
responding_filter1='J'
responding_filter2='H'
responding_filter3='Ks'


#list of sources, in ra and dec
source_list='../stat/lowz_ws_type1_Y_list_COSMOS.txt'

lc_path='../lowz_lc/final_catalog_well_sampled_type1_lc_Aug2018/'+field+'/'

jav_stat_path='../javelin_multiband_stat/'

out_file_name=jav_stat_path+'results_lowz_ws_type1_'+field+'_'+driving_filter+'_vs_'+responding_filter1+'_'+responding_filter2+'_and_'+responding_filter3+'.txt'


n_cores=4

dat=np.loadtxt(source_list,dtype='str').transpose()
ra=dat[0].astype(np.float)
dec=dat[1].astype(np.float)

#ra=ra[0:3]
#dec=dec[0:3]

################################################################################
#we define a function to calculate the cross correlation per every source, and estimate the errors using Javelin
def do_javelin(ra,dec):
    #ra and dec are converted to 6 decimals format
    ras="%.6f" % ra
    decs="%.6f" % dec

    print "########## computing Javelin for source located at ra=%f and dec=%f  ##########" % (ra, dec)

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

        lcd_name='temp/driving_lc_'+driving_filter+'_'+str(ras)+'_'+str(decs)+'.txt'
        np.savetxt(lcd_name,np.transpose([jd_driving/(1.0+zspec_driving),flux_driving,errflux_driving]))

        try:

            #reading the responding filter 1 data
            if responding_filter1!='Q':
                agn_responding=lc_path+responding_filter1+'/agn_'+str(ras)+'_'+str(decs)+'_'+responding_filter1+'.fits'
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
                    agn_driving=lc_path+responding_filter1+'/bin3_onechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                except:
                    agn_driving=lc_path+responding_filter1+'/bin3_morechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                jd_0=2455000
                head_responding=arch_responding[0].header
                datos_responding=arch_responding[1].data
                jd_responding=datos_responding['JD']-jd_0
                flux_responding=datos_responding['fluxQ']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
                errflux_responding=datos_responding['errfluxQ']*1e27
                zspec_responding=head_responding['REDSHIFT']

            #converting lcs into a format accepted by the fortran method
            lcr1_name='temp/responding_lc_'+responding_filter1+'_'+str(ras)+'_'+str(decs)+'.txt'
            np.savetxt(lcr1_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))


            #reading the responding filter 2 data
            if responding_filter2!='Q':
                agn_responding=lc_path+responding_filter2+'/agn_'+str(ras)+'_'+str(decs)+'_'+responding_filter2+'.fits'
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
                    agn_driving=lc_path+responding_filter2+'/bin3_onechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                except:
                    agn_driving=lc_path+responding_filter2+'/bin3_morechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                jd_0=2455000
                head_responding=arch_responding[0].header
                datos_responding=arch_responding[1].data
                jd_responding=datos_responding['JD']-jd_0
                flux_responding=datos_responding['fluxQ']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
                errflux_responding=datos_responding['errfluxQ']*1e27
                zspec_responding=head_responding['REDSHIFT']

            #converting lcs into a format accepted by the fortran method
            lcr2_name='temp/responding_lc_'+responding_filter2+'_'+str(ras)+'_'+str(decs)+'.txt'
            np.savetxt(lcr2_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))


            #reading the responding filter 3 data
            if responding_filter3!='Q':
                agn_responding=lc_path+responding_filter3+'/agn_'+str(ras)+'_'+str(decs)+'_'+responding_filter3+'.fits'
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
                    agn_driving=lc_path+responding_filter3+'/bin3_onechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                except:
                    agn_driving=lc_path+responding_filter3+'/bin3_morechip_'+str(ras)+'_'+field+'.fits'
                    arch_responding=pf.open(agn_responding)
                jd_0=2455000
                head_responding=arch_responding[0].header
                datos_responding=arch_responding[1].data
                jd_responding=datos_responding['JD']-jd_0
                flux_responding=datos_responding['fluxQ']*1e27 #the flux value is multiplicated by 1e27 to avoid numerical errors produced by small numbers
                errflux_responding=datos_responding['errfluxQ']*1e27
                zspec_responding=head_responding['REDSHIFT']

            #converting lcs into a format accepted by the fortran method
            lcr3_name='temp/responding_lc_'+responding_filter3+'_'+str(ras)+'_'+str(decs)+'.txt'
            np.savetxt(lcr2_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))




            #running Javelin


            cont=get_data([lcd_name],names=[driving_filter])
            cmod=Cont_Model(cont)
            cmod.do_mcmc(nwalkers=100, nburn=100, nchain=200, fchain=jav_stat_path+"chain_cont_"+driving_filter+"_vs_"+responding_filter1+"_"+responding_filter2+"_and_"+responding_filter3+"_"+str(ras)+"_"+str(decs)+".txt")

            bothdata=get_data([lcd_name,lcr1_name,lcr2_name,lcr2_name],names=[driving_filter,responding_filter1,responding_filter2,responding_filter3])
            laglimit = [[cent_lowlimit, cent_uplimit],[cent_lowlimit, cent_uplimit],[cent_lowlimit, cent_uplimit]]
            mod_2band=Rmap_Model(bothdata)#Rmap_Model(bothdata)
            mod_2band.do_mcmc(nwalkers=100, nburn=100, nchain=200,conthpd=cmod.hpd,laglimit=laglimit,fchain=jav_stat_path+"jav_chain_all_"+driving_filter+"_vs_"+responding_filter1+"_"+responding_filter2+"_and_"+responding_filter3+"_"+str(ras)+"_"+str(decs)+".txt")

            sigma, tau, lag1, width1, scale1, lag2, width2, scale2, lag3, width3, scale3 =np.loadtxt(jav_stat_path+"jav_chain_all_"+driving_filter+"_vs_"+responding_filter1+"_"+responding_filter2+"_and_"+responding_filter3+"_"+str(ras)+"_"+str(decs)+".txt", unpack = True, usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            centau_median1 = np.median(lag1)
            centau_uperr1 = (stats.scoreatpercentile(lag1, perclim))-centau_median1
            centau_loerr1 = centau_median1-(stats.scoreatpercentile(lag1, (100.-perclim)))
            len_chain1=len(lag1[np.where(lag1>-2000000000000)])

            centau_median2 = np.median(lag2)
            centau_uperr2 = (stats.scoreatpercentile(lag2, perclim))-centau_median2
            centau_loerr2 = centau_median2-(stats.scoreatpercentile(lag2, (100.-perclim)))
            len_chain2=len(lag2[np.where(lag2>-2000000000000)])

            centau_median3 = np.median(lag3)
            centau_uperr3 = (stats.scoreatpercentile(lag3, perclim))-centau_median3
            centau_loerr3 = centau_median3-(stats.scoreatpercentile(lag3, (100.-perclim)))
            len_chain3=len(lag3[np.where(lag3>-2000000000000)])

            return (ra,dec,zspec_driving,len_chain1,centau_median1,centau_loerr1,centau_uperr1,len_chain2,centau_median2,centau_loerr2,centau_uperr2,len_chain3,centau_median3,centau_loerr3,centau_uperr3)

        except:
            print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO RESPONDING LCs available  ##########" % (ra, dec)
            cmd='rm '+lcd_name
            os.system(cmd)
            return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999)
    except:

        print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO DRIVING LC available  ##########" % (ra, dec)

        return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999)




################################################################################
#we run the cross correlation using multiprocessing

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return do_javelin(*args)

def cc_parallel(ncores,ra_array,dec_array,save_file):
    #creating the list used by multi_run_wrapper
    #arg_list=[]
    #for i in xrange(len(ra_array)):
    #    arg_list.append((ra_array[i],dec_array[i]))

    #pool = Pool(processes=ncores)
    #results = pool.map(multi_run_wrapper,arg_list)
    #pool.close()
    #pool.join()

    result_list=[]
    for i in xrange(len(ra_array)):
        #arg_list.append((ra_array[i],dec_array[i],True))
        result=do_javelin(ra_array[i],dec_array[i])
        result_list.append(result)

    results=np.array(result_list)

    head='ra  dec  zspec  len_chain_'+responding_filter1+'  lag_median_'+responding_filter1+'  lag_lowerr_'+responding_filter1+'  lag_uperr_'+responding_filter1+' len_chain_'+responding_filter2+'  lag_median_'+responding_filter2+'  lag_lowerr_'+responding_filter2+'  lag_uperr_'+responding_filter2+' len_chain_'+responding_filter3+'  lag_median_'+responding_filter3+'  lag_lowerr_'+responding_filter3+'  lag_uperr_'+responding_filter3
    np.savetxt(save_file,results,header=head)
    print "File %s writen" % (save_file)
    return (results)

################################################################################

cc_parallel(4,ra,dec,out_file_name)
#results=do_javelin(149.444580,2.119920)
#print results
