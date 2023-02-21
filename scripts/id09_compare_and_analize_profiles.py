# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from srxraylib.plot.gol import plot

""" This set of functions is used to analize the vertical profiles 
and to compare them with SHADOW and WOFRY1D simulations, there are other
functions to retrive the mirror surface profile using the vertical slit scans  """

f_size = 12

def read_csv(file_name, source='shadow'):

    """ This function reads CSV files from SHADOW and WOFRY1D simulations """

    df = pd.read_csv(file_name, sep=',|\s+', comment = '#', engine='python')
        
    x = np.array(df.iloc[:,0])
    
    if source == 'shadow' or source == 'wofry':

        y = np.array(df.iloc[:,1])/np.max(np.array(df.iloc[:,1]))

    elif source == 'exp':

        y = np.array(df.iloc[:,1])
    else:
        print('uknown file source')

    return x, y

def compare_plot(csv_files, shift=0.0, base_line = 0.11, x_lim = 600):

    """ This function compares the experimental data with simulations """

    for csv_file in csv_files:
    
        df = pd.read_csv(csv_file, sep=',|\s+', comment = '#', engine='python')
            
        x = np.array(df.iloc[:,0])   
    
        if "Energy" in csv_file:
    
            if len(re.findall(r'\d+', csv_file)) == 2:
                legend = "Energy scan: {} keV".format(re.findall(r'\d+\.\d+', csv_file)[0])
    
            elif len(re.findall(r'\d+', csv_file)) == 1:
                legend = "Energy scan: {} keV".format(re.findall(r'\d+', csv_file)[0])
            
            else:
                raise RuntimeError("ERROR: Energy of the scan was not found")
    
        elif "Curvature" in csv_file:
    
            if len(re.findall(r'\d+', csv_file)) == 2:
                legend = "Mirror curvature scan: {} km".format(re.findall(r'\d+\.\d+', csv_file)[0])
    
            elif len(re.findall(r'\d+', csv_file)) == 1:
                legend = "Mirror curvature scan: {} km".format(re.findall(r'\d+', csv_file)[0])
            
            else:
                raise RuntimeError("ERROR: Mirror curvature of the scan was not found")
    
        else:
            raise RuntimeError("ERROR: type of scan was not identified")    
        
    
        if 'exp' in csv_file:
    
            y = np.array(df.iloc[:,1])
            #rev_exp_y = y[::-1]
            #rev_exp_y = y
            #shift_exp_y = np.roll(rev_exp_y, shift) - base_line
            y -= base_line
            y /= np.max(y)
    
            x+=shift
    
            plt.plot(-x, y, label='measurement')
    
        elif 'wofry' in csv_file:
    
            y = np.array(df.iloc[:,1])/np.max(np.array(df.iloc[:, 1]))
            
            #rev_wofry_y = y[::-1]

            if 'ret' in csv_file:
                plt.plot(-x, y, label='wofry_retrived')
            else:
                plt.plot(x, y, label='wofry')
                   
            #x += shift            
    
        elif 'shadow' in csv_file:

            y = np.array(df.iloc[:,1])/np.max(np.array(df.iloc[:, 1]))

            if 'ret' in csv_file:               
                plt.plot(x, y, label='shadow_retrived')
            else:
                plt.plot(-x, y, label='shadow')    
        else:
            raise RuntimeError("ERROR: type of scan was not identified") 

    plt.ylabel("Normalized intensity [a.u.]", fontsize= f_size)    
    plt.xlabel("Vertical [$\mu$m]", fontsize= f_size)

    plt.xlim(-x_lim, x_lim)
    plt.ylim(-0.04, 1.01)      

    plt.legend()
    plt.title(legend)    
    plt.show()


def get_slope_profile(csv_file, save_file = False):

    """ This function was used to retrive the surface mirror profile using the
	file which has the peak position in function of the slit position """

    toro_pos = 44540
    toro_angle = 2.5e-3
    ss1vo_pos = 54359
    xeye_pos = 55229    

    pixel_size = 2.87    

    df_pos = pd.read_csv(csv_file, sep=',|\s+', comment = '#', engine='python')
        
    slit_pos = (np.array(df_pos.iloc[:, 0]) - 0.15) # in milimeters and centered and inverted   

    peak_pos_pixel = np.array(df_pos.iloc[:, 1]) # in pixels

    half_pixel_range = (peak_pos_pixel[0] - peak_pos_pixel[-1]) / 2 * pixel_size #in milimeters

    peak_pos_xeye = ((peak_pos_pixel * pixel_size) - ((peak_pos_pixel[0] * pixel_size) - half_pixel_range) ) * 1e-6   #in meters and centered    

    fit = np.polyfit(slit_pos, peak_pos_xeye, 1)  

    x_fit = np.linspace(-0.5, 0.5, 600)
    y_fit = x_fit * fit[0] + fit[1]

    print('Rad curv:', 1/fit[0])

    diff = peak_pos_xeye - (fit[0]*slit_pos + fit[1])        

    plot(slit_pos, peak_pos_xeye,
     x_fit, y_fit,
     slit_pos, diff,
     xtitle="Slit position [mm]",
     ytitle="Position on xeye [m]",
     legend=["measurements", "linear fit", "difference"],
     show=1
    )    

    slope = diff/(xeye_pos*1e-3 - ss1vo_pos*1e-3) * (-1) / 2 #from Manolo

    print("Slope error RMS: ", np.std(slope))
    
    fit_proj = np.polyfit([ss1vo_pos - toro_pos, xeye_pos - toro_pos], [slit_pos[-1], peak_pos_xeye[0]*1e3], 1)

    dummy_x  = np.linspace(0, 10800)

    plot([0, ss1vo_pos - toro_pos, xeye_pos - toro_pos], [fit_proj[1], slit_pos[-1], peak_pos_xeye[0]*1e3],
         dummy_x, dummy_x*fit_proj[0]+fit_proj[1], xtitle="Distance from the source [mm]", ytitle="Vertical axis [mm]",
         title="Beamline layout", marker=['o', ''], show=1)

    print(peak_pos_xeye[0] * 1e3)

    print(f"Slope: {fit_proj[0]} and intercept: {fit_proj[1]}")

    print("L sin (theta) is:", 2 * fit_proj[1])

    toro_peak_pos = ((fit_proj[1])/(slit_pos[0]) * slit_pos)/(np.sin(toro_angle))

    #toro_peak_pos = (2 * fit_proj[1]/np.sin(toro_angle))

    plot(-toro_peak_pos, -slope*1e6,
         xtitle="Slit position [mm]",
         ytitle="slope on mirror [urad]",
         title="retrieved slope error",
         show=1)


    plot(-toro_peak_pos, -np.cumsum(slope)*1e6,
         xtitle="Slit position [mm]",
         ytitle="height on mirror [nm]",
         title="retrieved height error",
         show=1)

    if save_file:

        df_height = pd.DataFrame({'position[mm]':-toro_peak_pos, 'slopes[urad]':-slope*1e6})
        df_height.to_csv('test_slope_profile.csv', index=False)
        print('file test_slope_profile.csv has been saved to disk')
    
    return slit_pos, diff


def compare_plot_slit(csv_slit_pos, csv_file_sim, factor=1, shift=0.0, base_line = 0.0):

    df_slit = pd.read_csv(csv_slit_pos, sep=',|\s+', comment = '#', engine='python')        
    slit_positions = np.array(df_slit.iloc[:, 0])
    peak_intensities = np.array(df_slit.iloc[:, 1])

    slit_positions *= factor
    peak_intensities -= base_line
    peak_intensities/= np.max(peak_intensities)
    

    df_sim = pd.read_csv(csv_file_sim, sep=',|\s+', comment = '#', engine='python')
    x = np.array(df_sim.iloc[:, 0])    

    if 'wofry' in csv_file_sim:
        x -= shift
        y = np.array(df_sim.iloc[:, 1])/np.max(np.array(df_sim.iloc[:, 1]))
        plt.plot(-x, y, label='wofry')

    elif 'shadow' in csv_file_sim:
        x += shift
        y = np.array(df_sim.iloc[:, 1])/np.max(np.array(df_sim.iloc[:, 1]))
        plt.plot(x, y, label='shadow')
    else:
        raise RuntimeError("ERROR: type of simulations code was not identified")        
    

    plt.plot(-slit_positions*1e3, peak_intensities, label='From slit scan')    
    plt.xticks(fontsize= f_size)
    plt.yticks(fontsize= f_size)

    plt.legend(fontsize= f_size)
    plt.xlabel("Z [um]", fontsize= f_size)
    
    plt.show()

if __name__=="__main__":    
    #pass

    #fit = get_slope_profile('slit_pos_vs_peak_pos.csv')
    
    #compare_pixel_slit('v_profile_slit_pos.csv', 'v_profile_pixels_selected.csv' , pixel_size=2.99, shift=0.289)    
    
    
    #compare_plot('Energy_scan_ver_prof_17.35_keV_exp.csv', shift=21, base_line = 0.1)
    #compare_plot('Energy_scan_ver_prof_17.35_keV_shadow_mod.csv')
    #compare_plot('Energy_scan_ver_prof_17.35_keV_shadow_modd.csv')
    #compare_plot('Energy_scan_ver_prof_17.35_keV_shadow_m.csv')
    #compare_plot('Energy_scan_ver_prof_17.35_keV_wofry.dat')   

    #compare_plot('Energy_scan_ver_prof_17.95_keV_exp.csv', shift=6, base_line = 0.1)
    #compare_plot('Energy_scan_ver_prof_17.95_keV_shadow.csv')
    #compare_plot('Energy_scan_ver_prof_17.95_keV_wofry.dat')  

    #compare_plot('Energy_scan_ver_prof_18.49_keV_exp.csv', shift=6, base_line = 0.05)
    #compare_plot('Energy_scan_ver_prof_18.49_keV_shadow.csv')
    #compare_plot('Energy_scan_ver_prof_18.49_keV_wofry.dat')  

    #compare_plot('Energy_scan_ver_prof_17.8_keV_exp.csv', shift=0.0, base_line = 0.3)
    #compare_plot('Energy_scan_ver_prof_17.8_keV_shadow.csv')
    #compare_plot('Energy_scan_ver_prof_17.8_keV_wofry.dat')

    #compare_plot('Energy_scan_ver_prof_18.04_keV_exp.csv', shift=90, base_line = 0.3) #90
    #compare_plot('Energy_scan_ver_prof_18.04_keV_shadow_yaw.csv')

    #compare_plot('Energy_scan_ver_prof_18.04_keV_exp.csv', shift=52, base_line = 0.3) #90
    #compare_plot('Energy_scan_ver_prof_18.04_keV_shadow_last.csv')

    #compare_plot('Curvature_scan_ver_prof_9_km_exp.csv', shift=128, base_line = 0.03)
    #compare_plot('Curvature_scan_ver_prof_9_km_shadow.csv')
    #compare_plot('Curvature_scan_ver_prof_9_km_wofry.dat', shift=40)

    #compare_plot('Curvature_scan_ver_prof_7.3_km_exp.csv', shift=118, base_line = 0.03)
    #compare_plot('Curvature_scan_ver_prof_7.3_km_shadow.csv')
    #compare_plot('Curvature_scan_ver_prof_7.3_km_wofry.dat', shift=40)

    #compare_plot('Curvature_scan_ver_prof_8_km_exp.csv', shift=15, base_line = 0.07)
    #compare_plot('Curvature_scan_ver_prof_8_km_shadow_pink.csv')
    #compare_plot('Curvature_scan_ver_prof_8_km_shadow_pink_m.csv')

    #compare_plot('Curvature_scan_ver_prof_8_km_exp.csv', shift=0.0, base_line = 0.03)
    #compare_plot('Curvature_scan_ver_prof_8_km_shadow_m.csv')
    #compare_plot('Curvature_scan_ver_prof_8_km_wofry.dat')

    #compare_plot(('Curvature_scan_ver_prof_20_km_exp.csv', 'Curvature_scan_ver_prof_20_km_shadow_mod_prof.csv'), shift=0.0, base_line = 0.03)    
    #compare_plot(('Curvature_scan_ver_prof_20_km_exp.csv', 'Curvature_scan_ver_prof_20_km_shadow.csv'), shift=0.0, base_line = 0.03)

    #compare_plot('Energy_scan_ver_prof_18.28_keV_exp.csv', shift=21, base_line = 0.1)
    #compare_plot('Energy_scan_ver_prof_18.28_keV_shadow.csv') 

    pass