#code to run calculate zdcf, using the code plike to measure the lag errors.
#the code can be used to calculate the autocorrelation function
import numpy as np
import os
import astropy.io.fits as pf
from multiprocessing import Pool

################################################################################
#cross correlation parameters:
auto_corr=False

lag_lowlimit=-400
lag_uplimit=900

################################################################################
################################################################################
#file parameters

field='COSMOS' #COSMOS, XMM_LSS, ELAIS_S1, ECDFS

#auto-corr parameters:
ac_path='../../KIAA_work/ac_stat/'
ac_file='../../KIAA_work/ac_stat/QUEST_SDSS_ac_results_'+field+'.txt'


#filters used
driving_filter='Q'
responding_filter='Y'

#list of sources, in ra and dec
source_list='../stat/lowz_ws_type1_'+driving_filter+'_list_COSMOS.txt'
#source_list='../../KIAA_work/stat/QUEST_SDSS_lc_well_sampled_'+field+'.txt'

lc_path='../lowz_lc/final_catalog_well_sampled_type1_lc_Aug2018/'+field+'/'
#lc_path='../../KIAA_work/QUEST_SDSS_sample_lc/'

out_file_name='../zdcf_stat_Aug2018/results_lowz_ws_type1_'+field+'_'+driving_filter+'_'+responding_filter+'.txt'


n_cores=2

dat=np.loadtxt(source_list,dtype='str').transpose()
ra=dat[0].astype(np.float)
dec=dat[1].astype(np.float)



################################################################################
#we define a function to calculate the cross correlation per every source, and estimate the errors
def do_zdcf(ra,dec):
    #ra and dec are converted to 6 decimals format
    ras="%.6f" % ra
    decs="%.6f" % dec

    print "########## computing zdcf for source located at ra=%f and dec=%f  ##########" % (ra, dec)

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
            if auto_corr:
                try:
                    agn_driving=lc_path+'bin3_onechip_'+str(ras)+'_'+str(decs)+'_'+field+'.fits'
                    arch_driving=pf.open(agn_driving)
                except:
                    agn_driving=lc_path+'bin3_morechip_'+str(ras)+'_'+str(decs)+'_'+field+'.fits'
                    arch_driving=pf.open(agn_driving)
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
            try: zspec_driving=head_driving['REDSHIFT']
            except: zspec_driving=head_driving['ZSPEC']



        print "redhisft", zspec_driving

        #converting lcs into a format accepted by the fortran method
        lcd_name='driving_lc_'+driving_filter+'_'+str(ras)+'_'+str(decs)+'.txt'
        np.savetxt(lcd_name,np.transpose([jd_driving/(1.0+zspec_driving),flux_driving,errflux_driving]))



        #for the case of auto-correlation
        if auto_corr:
            try:
                #running zdcf (v2.2)
                print "running AC"
                cmd='./zdcf.exe << EOF \n1 \nAC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+' \nn \n11 \nn \n100 \n'+lcd_name+' \n EOF'
                os.system(cmd)

                cmd='rm '+lcd_name
                os.system(cmd)

                #loading AC results
                aux=np.genfromtxt('AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.dcf')
                lag=aux[:,0]
                lag_lowerr=aux[:,1]
                lag_uperr=aux[:,2]
                ac_val=aux[:,3]-1.0/np.exp(1.0)
                ac_lowerr=aux[:,4]
                ac_uperr=aux[:,5]

                print ac_val

                zero_crossings = np.where(np.diff(np.sign(ac_val)))[0]
                print zero_crossings
                first_zc = zero_crossings[0]
                lag_before_zc = lag[first_zc]
                lag_after_zc = lag[first_zc+1]

                ac_before_zc = lag[first_zc]
                ac_after_zc = lag[first_zc+1]

                lag_zero = np.interp(0,np.array([ac_before_zc,ac_after_zc]),np.array([lag_before_zc,lag_after_zc]))

                #moving zdcf ".dcf" output to zdcf_stat
                cmd='mv AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.dcf '+ac_path
                os.system(cmd)

                #deleting other zdcf outputs
                cmd='rm AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.lc1'
                os.system(cmd)

                cmd='rm AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.lc2'
                os.system(cmd)

                return (ra,dec,lag_before_zc,lag_after_zc, lag_zero)

            except:

                print "########## computing AC with zdcf FAILS for source located at ra=%f and dec=%f  ##########" % (ra, dec)
                cmd='mv AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.dcf '+ac_path
                os.system(cmd)

                #deleting other zdcf outputs
                cmd='rm AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.lc1'
                os.system(cmd)

                cmd='rm AC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'.lc2'
                os.system(cmd)
                return (ra,dec,-9999,-9999,-9999)

        #for the case of cross-correlation
        else:

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




                #converting lcs into a format accepted by the fortran method
                lcr_name='responding_lc_'+responding_filter+'_'+str(ras)+'_'+str(decs)+'.txt'
                np.savetxt(lcr_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))

                #running zdcf (v2.2)
                cmd='./zdcf.exe << EOF \n2 \nCC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+' \nn \n11 \nn \n100 \n'+lcd_name+' \n'+lcr_name+' \n EOF'
                os.system(cmd)

                cmd='rm '+lcd_name+' '+lcr_name
                os.system(cmd)

                #running plike (v4.0)
                cmd='./plike_Paula.exe << EOF \nCC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf \n'+str(lag_lowlimit)+' \n'+str(lag_uplimit)+' \n EOF'
                os.system(cmd)

                try:
                    #loading plike results
                    aux=np.genfromtxt('CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf.plike.out')
                    lag_peak=aux[0]
                    lag_lowerr=aux[1]
                    lag_uperr=aux[2]

                    #loading zdcf results to obtain r in the peak
                    aux2=np.loadtxt('CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf',dtype='str').transpose()
                    lags=aux2[0].astype(np.float)
                    rs=aux2[3].astype(np.float)

                    r_peak=rs[np.where(lags==lag_peak)][0]
                except:
                    print "problems computing peak from plike for source located at ra=%f and dec=%f"
                    lag_peak=-8888
                    lag_lowerr=-8888
                    lag_uperr=-8888
                    lags=-8888
                    rs=-8888
                    r_peak=-8888

                #deleting plike.out
                cmd='rm CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf.plike.out'
                os.system(cmd)

                #moving zdcf ".dcf" output to zdcf_stat
                cmd='mv CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.dcf  ../zdcf_stat_Aug2018'
                os.system(cmd)

                #deleting other zdcf outputs
                cmd='rm CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.lc1'
                os.system(cmd)

                cmd='rm CC_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_'+responding_filter+'.lc2'
                os.system(cmd)



                return (ra,dec,zspec_driving,lag_peak,lag_lowerr,lag_uperr,r_peak)

            except:

                print "########## computing zdcf FAILS for source located at ra=%f and dec=%f, NO RESPONDING LC available  ##########" % (ra, dec)
                cmd='rm '+lcd_name
                os.system(cmd)
                return (ra,dec,-9999,-9999,-9999,-9999,-9999)

    except:

        print "########## computing zdcf FAILS for source located at ra=%f and dec=%f, NO DRIVING LC available  ##########" % (ra, dec)

        return (ra,dec,-9999,-9999,-9999,-9999,-9999)

################################################################################
#we run the cross correlation using multiprocessing

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return do_zdcf(*args)

def zdcf_parallel(ncores,ra_array,dec_array,save_file):
    #creating the list used by multi_run_wrapper
    arg_list=[]
    for i in xrange(len(ra_array)):
        arg_list.append((ra_array[i],dec_array[i]))

    pool = Pool(processes=ncores)
    results = pool.map(multi_run_wrapper,arg_list)
    pool.close()
    pool.join()

    #print results

    if auto_corr:
        head='ra  dec  lag_before_zc  lag_after_zc  lag_zero'
    else: head='ra dec zspec peak_zdcf  peak_lowerr  peak_uperr  r_peak'
    np.savetxt(save_file,results,header=head)
    print "File %s writen" % (save_file)
    return (results)

################################################################################
#zdcf_parallel(n_cores,ra,dec,ac_file)
zdcf_parallel(n_cores,ra,dec,out_file_name)
#results=do_zdcf(ra[0],dec[0])
#print results
