import numpy as np
#from uncertainties import ufloat_fromstr,ufloat,umath
import pandas as pd
#import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
from kelly_regression import *

#input_file='../stat/table_summary_spec_properties_var_features_logvalues_withBHmass_withL5100_reduced_noCIV_EW_DR14_DR12_DR7_18Feb27.csv'
input_file='../stat/table_summary_spec_properties_var_features_logvalues_correctedBHmass_withBHmass_withL5100_DR14_DR12_DR7_18March7.csv'
#input_file='../stat/table_summary_spec_properties_var_features_logvalues_withBHmass_withL5100_spec_properties_MgII_Hbeta_DR14_DR12_DR7_18Feb27.csv'


df = pd.read_csv(input_file)


def run_regression(input_file,xvars,yvar,plot_var,plot_zrange,zmin,zmax,plot_gamma_range,gmin,gmax,which_line):
    df = pd.read_csv(input_file)

    df['log(1+z)_err']=0.001

    if which_line=='all':
        df=df
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='not-CIV':
        df=df[(df['final_mass_line']!="CIV")]

        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='Halpha-Hbeta':
        df=df[((df['logFWHM_Halpha']>0) | (df['logFWHM_Hbeta']>0) )]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='MgII-Hbeta':
        df=df[((df['logFWHM_MgII']>0) & (df['logFWHM_Hbeta']>0) )]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='MgII-CIV':
        df=df[((df['logFWHM_MgII']>0) & (df['logFWHM_CIV']>0) )]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='Halpha':
        df=df[df['logFWHM_Halpha'].notnull()]
        df=df[df['logLline_Halpha'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='Hbeta':
        df=df[df['logFWHM_Hbeta'].notnull()]
        df=df[df['logLline_Hbeta'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='MgII':
        df=df[(df['final_mass_line']=="Hbeta+MgII") | (df['final_mass_line']=="MgII")]
        #df=df[df['logFWHM_MgII'].notnull()]
        #df=df[df['logLline_MgII'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='MgII2':
        #df=df[(df['final_mass_line']=="Hbeta+MgII") | (df['final_mass_line']=="MgII")]
        df=df[df['logFWHM_MgII'].notnull()]
        df=df[df['logLline_MgII'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='CIV':
        df=df[(df['final_mass_line']=="CIV")]
        #df=df[df['logFWHM_CIV'].notnull()]
        #df=df[df['logFWHM_CIV']>3.0]
        #df=df[df['logLline_CIV'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)

    elif which_line=='CIV2':
        #df=df[(df['final_mass_line']=="CIV")]
        df=df[df['logFWHM_CIV'].notnull()]
        df=df[df['logFWHM_CIV']>3.0]
        df=df[df['logLline_CIV'].notnull()]
        print "number of sources in the line group", which_line, " : ", len(df[yvar].values)


    df=df[(df['num_epochs'] >= 40) & (df['time_range_rest']>=200.0)]

    print "number of sources with good sampling: ",  len(df[yvar].values)

    #filter by variability
    if plot_var:
        df=df[(df['p_var'] >= 0.95) & ((df['ex_var']-df['ex_var_err'])>0.0)]

    print "number of variable sources: ", len(df[yvar].values)
    #filter by redshift
    if plot_zrange:
        df=df[(df['zspec'] >= zmin) & (df['zspec'] <= zmax)]

    print "final number of variable sources: ", len(df[yvar].values)

    #filter by gamma
    if plot_gamma_range:
        df=df[(df['loggamma_mcmc'] >= gmin) & (df['loggamma_mcmc']<=gmax)]

        print "number of sources in gamma range: ", len(df[yvar].values)

    nx=len(xvars)

    x = np.array(df[xvars])
    x = sm.add_constant(x)

    if yvar+'_err' in df.columns: results = sm.WLS(endog=df[yvar],exog=x,weights=1/(df[yvar+'_err']**2)).fit()
    else: results = sm.GLS(endog=df[yvar],exog=x).fit()
    print results.summary()

    if nx==1:

        if xvars[0]+'_err' in df.columns:
            x1err=df[xvars[0]+'_err'].values
            x1=df[xvars[0]].values
            print xvars[0], "has error"
        else:
            x1=df[xvars[0]].values
            x1err=np.zeros(len(x1))


        if yvar+'_err' in df.columns:
            yerr=df[yvar+'_err'].values
            y=df[yvar].values
            print yvar, "has error"

        else:
            y=df[yvar].values
            yerr=np.zeros(len(y))

        #aux=mcmc_regression_1x(x1, y, x1err=x1err, x2err=yerr, start=(1.,1.,0.5),starting_width=0.01, logify=False,nsteps=5000, nwalkers=100, nburn=100, output=(50,16,84))
        aux=linmix_err(x1, y, x1err=x1err, x2err=yerr)
        intercept=aux[0]
        slope=aux[1]
        scatter=aux[2]
        '''
        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[0]-intercept[1], intercept[2]-intercept[0]
        print "slope "+xvars[0]+": ", slope[0], slope[0]-slope[1], slope[2]-slope[0]
        print "scatter: ",scatter[0], scatter[0]-scatter[1], scatter[2]-scatter[0]
        '''

        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[1]
        print "slope "+xvars[0]+": ", slope[0], slope[1]
        print "scatter: ",scatter[0], scatter[1]

    elif nx==2:

        if xvars[0]+'_err' in df.columns:
            x1err=df[xvars[0]+'_err'].values
            x1=df[xvars[0]].values
            print xvars[0], "has error"
        else:
            x1=df[xvars[0]].values
            x1err=np.zeros(len(x1))

        if xvars[1]+'_err' in df.columns:
            x2err=df[xvars[1]+'_err'].values
            x2=df[xvars[1]].values
            print xvars[1], "has error"
        else:
            x2=df[xvars[1]].values
            x2err=np.zeros(len(x2))


        if yvar+'_err' in df.columns:
            yerr=df[yvar+'_err'].values
            y=df[yvar].values
            print yvar, "has error"

        else:
            y=df[yvar].values
            yerr=np.zeros(len(y))


        xarr,exarr=input_mlinmix_2var(x1,x2,x1err, x2err)
        aux=mlinmix_2x_err(xarr, y, x1err=exarr, x2err=yerr)
        #aux=mcmc_regression_2x(x1,x2, y, x1err=x1err, x2err=x2err, x3err=yerr, start=(1.,1.,1.,0.5),starting_width=0.01, logify=False,nsteps=5000, nwalkers=100, nburn=100, output=(50,16,84))

        print aux
        intercept=aux[0]
        slope1=aux[1]
        slope2=aux[2]
        scatter=aux[3]

        '''
        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[0]-intercept[1], intercept[2]-intercept[0]
        print "slope "+xvars[0]+": ", slope1[0], slope1[0]-slope1[1], slope1[2]-slope1[0]
        print "slope "+xvars[1]+": ", slope2[0], slope2[0]-slope2[1], slope2[2]-slope2[0]
        print "scatter: ",scatter[0], scatter[0]-scatter[1], scatter[2]-scatter[0]
        '''

        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[1]
        print "slope "+xvars[0]+": ", slope1[0], slope1[1]
        print "slope "+xvars[1]+": ", slope2[0], slope2[1]
        print "scatter: ",scatter[0], scatter[1]

    elif nx==3:

        if xvars[0]+'_err' in df.columns:
            print xvars[0], "has error"
            x1err=df[xvars[0]+'_err'].values
            x1=df[xvars[0]].values
        else:
            x1=df[xvars[0]].values
            x1err=np.zeros(len(x1))

        if xvars[1]+'_err' in df.columns:
            print xvars[1], "has error"
            x2err=df[xvars[1]+'_err'].values
            x2=df[xvars[1]].values
        else:
            x2=df[xvars[1]].values
            x2err=np.zeros(len(x2))

        if xvars[2]+'_err' in df.columns:
            print xvars[2], "has error"
            x3err=df[xvars[2]+'_err'].values
            x3=df[xvars[2]].values
        else:
            x3=df[xvars[2]].values
            x3err=np.zeros(len(x3))


        if yvar+'_err' in df.columns:
            print yvar, "has error"
            yerr=df[yvar+'_err'].values
            y=df[yvar].values

        else:
            y=df[yvar].values
            yerr=np.zeros(len(y))


	xarr,exarr=input_mlinmix_3var(x1,x2,x3,x1err,x2err,x3err)
        aux=mlinmix_3x_err(xarr, y, x1err=exarr, x2err=yerr)
        #aux=mcmc_regression_3x(x1,x2,x3, y, x1err=x1err, x2err=x2err, x3err=x3err, x4err=yerr, start=(1.,1.,1.,1.,0.5),starting_width=0.01, logify=False,nsteps=500, nwalkers=100, nburn=100, output=(50,16,84))
        intercept=aux[0]
        slope1=aux[1]
        slope2=aux[2]
        slope3=aux[3]
        scatter=aux[4]

        '''
        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[0]-intercept[1], intercept[2]-intercept[0]
        print "slope "+xvars[0]+": ", slope1[0], slope1[0]-slope1[1], slope1[2]-slope1[0]
        print "slope "+xvars[1]+": ", slope2[0], slope2[0]-slope2[1], slope2[2]-slope2[0]
        print "slope "+xvars[2]+": ", slope3[0], slope3[0]-slope3[1], slope3[2]-slope3[0]
        print "scatter: ",scatter[0], scatter[0]-scatter[1], scatter[2]-scatter[0]
        '''

        print "linear regression for ", yvar
        print "intercept: ", intercept[0], intercept[1]
        print "slope "+xvars[0]+": ", slope1[0], slope1[1]
	print "slope "+xvars[1]+": ", slope2[0], slope2[1]
	print "slope "+xvars[2]+": ", slope3[0], slope3[1]
        print "scatter: ",scatter[0], scatter[1]




    return aux



run_regression(input_file,['log(1+z)','logL5100_reduced44','logBHmass_reduced44'],'logA_mcmc',True,False,1.9,2.0,True,np.log10(0.5),np.log10(0.7),'not-CIV')
print "\n"
