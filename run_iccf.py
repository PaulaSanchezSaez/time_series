#code to run calculate iccf, using the code lagerror8 to measure the lag errors.

import numpy as np
import os
import astropy.io.fits as pf
from multiprocessing import Pool

################################################################################
#cross correlation parameters:

dt_tau=5 #days
cent_lowlimit=-200
cent_uplimit=200

change_maximum_shift='y'
desired_maximum_shift=200
grid_spacing=str(dt_tau)
minimum_correlation_coefficient=0.5
centroid_calculation_level=0.8
number_of_trials=10000

################################################################################
#file parameters

field='COSMOS' #COSMOS, XMM_LSS, ELAIS_S1, ECDFS

#filters used
driving_filter='Q'
responding_filter='Y'

#list of sources, in ra and dec
source_list='../stat/lowz_ws_type1_Q_list_COSMOS.txt'

lc_path='../lowz_lc/final_catalog_well_sampled_type1_lc/'+field+'/'

out_file_name='../iccf_stat/results_lowz_ws_type1_'+field+'_'+driving_filter+'_'+responding_filter+'_dt'+str(dt_tau)+'.txt'


n_cores=4

dat=np.loadtxt(source_list,dtype='str').transpose()
ra=dat[0].astype(np.float)
dec=dat[1].astype(np.float)

################################################################################
#Paulina's code to do the interpolated cross correlation with the original lcs

def do_ccros(data0,datai,tunit,output):
    #function to calculate the cross correlation index for different lags, with a time resolution of tunit
    #data0: driving lc, datai: responding lc, tunit: dt for the lags definition, output: file name with the results

    out = open(output, 'w')

    lc1 = data0
    lc2 = datai

    #the span between the lcs is calculated
    ta = np.max([lc1[0][0],lc2[0][0]])
    tb = np.min([lc1[0][-1],lc2[0][-1]])
    span = tb-ta
    shiftmx = span
    shiftmn = -span
    safe = 0.1*float(tunit)

    #the cross correlation is calculated with the epochs of the first lc (iccfa).
    tau = shiftmn
    tlaga, iccfa, nptsa = [],[],[]

    while tau <= shiftmx+safe:
        r,kount = do_loop(tau,1,lc1,lc2)
        tlaga.append(tau)
        iccfa.append(r)
        nptsa.append(kount)
        tau = tau+float(tunit)

    #the cross correlation is calculated with the epochs of the second lc (iccfb).
    tau = shiftmn
    tlagb, iccfb, nptsb = [],[],[]

    while tau <= shiftmx+safe:
        r,kount = do_loop(tau,-1,lc2,lc1)
        tlagb.append(tau)
        iccfb.append(r)
        nptsb.append(kount)
        tau = tau+float(tunit)

    #the results are saved in the output file, the final cross correlation index is calculated as the average of iccfa and iccfb.
    tau_out=[]
    iccf_out=[]
    for i in range(len(iccfa)):
        r_fin=(iccfa[i]+iccfb[i])/2.0
        tau_out.append(tlaga[i])
        iccf_out.append(r_fin)
        print >> out, tlaga[i],r_fin

    tau_out=np.array(tau_out)
    iccf_out=np.array(iccf_out)

    return (tau_out,iccf_out)


def do_loop(tau, switch, series1, series2):
    #function to calculate r for a given value of tau
    #tau: time lag, switch: needed to say if series1 is the driving lc (switch=1) or if series2 is the driving lc (switch=-1)
    #series1, series2: lcs

    xsum,xsum2,ysum,ysum2,xysum,kount = 0,0,0,0,0,0

    for i in range(len(series1[0])):
        time = series1[0][i]+switch*tau
        if time < series2[0][0] or time > series2[0][-1]:
            continue
        yval = np.interp(time,series2[0],series2[1])#interpolate(time,series2[0],series2[1])
        xsum = xsum+series1[1][i]
        xsum2 = xsum2+series1[1][i]**2
        ysum = ysum+yval
        ysum2 = ysum2+yval**2
        xysum = xysum+series1[1][i]*yval
        kount += 1

    rd1 = np.sqrt(xsum2*kount-xsum**2)
    rd2 = np.sqrt(ysum2*kount-ysum**2)
    if rd1 == 0 or rd2 == 0: r = 0
    else: r = (xysum*kount-xsum*ysum)/(rd1*rd2)

    return r,kount

def interpolate(x, xs, ys):
    #function to interpolate the light curve and evaluate it at a given timeself.
    #x: where the interpolated function is evaluated, xs: jd of the time series, ys: flux of the time sieries.

    for i in range(len(xs)-1):
        if (xs[i] < x <= xs[i+1]):
            ny = (x*(ys[i+1]-ys[i])+xs[i+1]*ys[i]-xs[i]*ys[i+1])/(xs[i+1]-xs[i])
    return ny

################################################################################
#Paulina's code to estimate the peak and the centroid of the ICCF

def get_shifts(lags, coef, lowlimit, uplimit):

    shifts = []

    #the peak is estimated
    peak = -100
    ipeak = 0
    for i in range(len(lags)):
        if lowlimit < lags[i] < uplimit and coef[i] > peak:
            ipeak = i
            peak = coef[i]
            tau_peak=lags[i]
    #the centroid is calculated
    cut = centroid_calculation_level*peak #we only considered values higher than 0.8*peak

    #the right limit of the peak is calculated
    for j in range(ipeak,len(coef)):
        if coef[j] > cut:
            continue
        else:
            jright = j
            break

    #the left limit of the peak is calculated
    for j in reversed(range(ipeak)):
        if coef[j] > cut:
            continue
        else:
            jleft = j
            break

    #the centroid is calculated, a minimum of 3 points is requested to calculate it.
    sum1, sum2 = 0, 0
    peak_range = range(jleft,jright-1)
    if len(peak_range) > 3:
        for j in peak_range:
            sum1 = sum1 + coef[j]*lags[j]
            sum2 = sum2 + coef[j]
        centroid = sum1/sum2
    else: centroid = 0


    return (tau_peak,centroid)



################################################################################
#we define a function to calculate the cross correlation per every source, and estimate the errors
def do_iccf(ra,dec):
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

        lcd_name='temp/driving_lc_'+driving_filter+'_'+str(ras)+'_'+str(decs)+'.txt'
        np.savetxt(lcd_name,np.transpose([jd_driving/(1.0+zspec_driving),flux_driving,errflux_driving]))

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
            lcr_name='temp/responding_lc_'+responding_filter+'_'+str(ras)+'_'+str(decs)+'.txt'
            np.savetxt(lcr_name,np.transpose([jd_responding/(1.0+zspec_driving),flux_responding,errflux_responding]))


            #calculate the iccf from the original lc
            data0=np.array([jd_driving/(1.0+zspec_driving),flux_driving])
            datai=np.array([jd_responding/(1.0+zspec_driving),flux_responding])

            tau_org,iccf_org=do_ccros(data0,datai,dt_tau,'../iccf_stat/org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt')
            peak_org,centroid_org=get_shifts(tau_org,iccf_org, cent_lowlimit, cent_uplimit)
            print "********* peak_org=%f  cent_org=%f  *********" % (peak_org,centroid_org)

            #calculate the peak and centroid distribution from FR/RSS
            #we choose the desired maximum shift:
            if change_maximum_shift=='y':
                cmd='./iccf_Paula.exe << EOF \n3 \n'+lcd_name+' \n'+lcr_name+' \n'+str(change_maximum_shift)+' \n'+str(desired_maximum_shift)+' \n'+str(grid_spacing)+' \n0 \n0 \n'+str(minimum_correlation_coefficient)+' \n'+str(centroid_calculation_level)+' \ny \n'+str(number_of_trials)+' \n EOF'
            #we use the measured maximum shift:
            else:
                cmd='./iccf_Paula.exe << EOF \n3 \n'+lcd_name+' \n'+lcr_name+' \n'+str(change_maximum_shift)+' \n'+str(grid_spacing)+' \n0 \n0 \n'+str(minimum_correlation_coefficient)+' \n'+str(centroid_calculation_level)+' \ny \n'+str(number_of_trials)+' \n EOF'
            os.system(cmd)

            #we read the outputs of lagerror8:

            #centroid distribution
            arch_cent=np.genfromtxt(lcd_name[0:10]+'.centdist.dat',usecols=(0,1))
            tau_cent=arch_cent[:,0]
            dist_cent=arch_cent[:,1]

            n_cent = len(dist_cent)
            sum_bot, sum_top = 0, 0
            for i in range(n_cent):
                if sum_bot <= 0.1359:
                    sum_bot = sum_bot + dist_cent[i]
                    lag_bot =  tau_cent[i]
                if sum_top <= 0.1359:
                    sum_top = sum_top + dist_cent[n_cent-i-1]
                    lag_top = tau_cent[n_cent-i-1]

            cent_lag_bot=lag_bot
            cent_lag_top=lag_top

            #peak distribution
            arch_peak=np.genfromtxt(lcd_name[0:10]+'.peakdist.dat',usecols=(0,1))
            tau_peak=arch_peak[:,0]
            dist_peak=arch_peak[:,1]

            n_peak = len(tau_peak)
            sum_bot, sum_top = 0, 0
            for i in range(n_peak):
                if sum_bot <= 0.1359:
                    sum_bot = sum_bot + dist_peak[i]
                    lag_bot =  tau_peak[i]
                if sum_top <= 0.1359:
                    sum_top = sum_top + dist_peak[n_peak-i-1]
                    lag_top = tau_peak[n_peak-i-1]

            peak_lag_bot=lag_bot
            peak_lag_top=lag_top



            cmd='mv '+lcd_name[0:10]+'.centdist.dat ../iccf_stat/centdist_org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt'
            os.system(cmd)
            cmd='mv '+lcd_name[0:10]+'.peakdist.dat ../iccf_stat/peakdist_org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt'
            os.system(cmd)
            cmd='mv '+lcd_name[0:10]+'.summary.txt ../iccf_stat/summary_org_iccf_'+str(ras)+'_'+str(decs)+'_'+driving_filter+'_vs_'+responding_filter+'.txt'
            os.system(cmd)

            return (ra,dec,zspec_driving,peak_org,peak_org-peak_lag_bot,peak_lag_top-peak_org,centroid_org,centroid_org-cent_lag_bot,cent_lag_top-centroid_org)

        except:
            print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO RESPONDING LC available  ##########" % (ra, dec)
            cmd='rm '+lcd_name
            os.system(cmd)
            return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999)
    except:

        print "########## computing iccf FAILS for source located at ra=%f and dec=%f, NO DRIVING LC available  ##########" % (ra, dec)

        return (ra,dec,-9999,-9999,-9999,-9999,-9999,-9999,-9999)

################################################################################
#we run the cross correlation using multiprocessing

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return do_iccf(*args)

def cc_parallel(ncores,ra_array,dec_array,save_file):
    #creating the list used by multi_run_wrapper
    arg_list=[]
    for i in xrange(len(ra_array)):
        arg_list.append((ra_array[i],dec_array[i]))

    pool = Pool(processes=ncores)
    results = pool.map(multi_run_wrapper,arg_list)
    pool.close()
    pool.join()

    head='ra dec zspec peak_org  peak_lowerr  peak_uperr  centroid_org  cent_lowerr  cent_uperr'
    np.savetxt(save_file,results,header=head)
    print "File %s writen" % (save_file)
    return (results)

################################################################################

cc_parallel(n_cores,ra,dec,out_file_name)
#results=do_iccf(149.666779,2.286390)
#print results
