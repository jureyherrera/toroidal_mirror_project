# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from silx.math.fit import FitManager
from silx.math.fit import fittheories
from silx.math.fit.functions import sum_lorentz, sum_gauss
from srxraylib.plot.gol import plot

""" This set of functions is used to extract the vertical profiles
from the h5 files resulting from the measurements """

def get_data(h5file):
    """ This function just to load data from the h5 and identifies
        the scan type """	

    h5_file = h5py.File(h5file, 'r')
    data = np.array(h5_file['data'])

    try:
        h5_file['info/motors_paramters/ss1vo']
        slit_positions = np.array(h5_file['positions'])
        print('Type of scan is slit scan')
        return data, slit_positions
        
    except:
        print('Type is not a slit scan')

    try:
        h5_file['info/motors_paramters/mono1e']
        energies = np.array(h5_file['positions'])
        print('Type of scan is energy')
        return data, energies
        
    except:
        print('Type of scan is mirror radius')

    try:
        h5_file['info/motors_paramters/m1bend']
        m_curvatures = np.array(h5_file['positions'])
        return data, m_curvatures
    except KeyError:
        print('Type of scan is not identify')
    h5_file.close()

def plot_img(data, position):

    """ Image plotter using a given pixel size, notice that position can be
	any step of energy or mirror radius """

    f_size = 12

    #pixel_size = 3.75/2 #pixel size in um 

    pixel_size = 2.87   

    height, width = data[position, :, :].shape

    x = np.linspace(-width/2, width/2, width) * pixel_size
    y = np.linspace(-height/2, height/2, height) * pixel_size	

    plt.pcolormesh(x, y, data[position, :, :], cmap=plt.cm.viridis, shading='auto')

    plt.ylabel("Vertical [$\mu$m]", fontsize= f_size)
    plt.xlabel("Horizontal [$\mu$m]", fontsize= f_size)
    plt.xticks(fontsize= f_size)
    plt.yticks(fontsize= f_size)

    plt.show()

def e_plot_profile(data, energies, position, normalized=True, save_file=False):

    """ Used for energy scans, this function plots the vertical profile, it can
	   save a CSV file with the energy step on the filename """
    
    try:
        label = f'profile for {energies[position]} keV'
        csv_name = f'Energy_scan_ver_prof_{energies[position]}_keV_exp.csv'
    except:
        print(f'Energy step {energies[position]} keV has not been found')    
    
    # values for the plotting #
    f_size = 12
    #pixel_size = 3.75/2 #pixel size in um
    #pixel_size = 2.73
    pixel_size = 2.87
    #pixel_size = 2.8
    origin_y = 570

    height, width = data[position, :, :].shape
    
    y = np.linspace(0 - origin_y, height - origin_y, num=height) * pixel_size

    #y = np.linspace(-height/2, height/2, height) * pixel_size

    v_profile = np.zeros_like(data[position, :, 0])	

    for i in range(width):
        v_profile = np.add(v_profile, data[position, :, i])

    if normalized:
        v_profile /= np.max(v_profile)
        plt.ylabel("Normilized intensity [a.u.]", fontsize= f_size)
    else:
        plt.ylabel("Intensity [a.u.]", fontsize= f_size)
    
    plt.plot(y, v_profile, label=label, linewidth=2)
    plt.xlabel("Vertical [$\mu$m]", fontsize= f_size)
        
    plt.xticks(fontsize= f_size)
    plt.yticks(fontsize= f_size)

    plt.legend(fontsize= f_size)

    plt.show()

    if save_file:

        df = pd.DataFrame({'vertical':y,'intensity':v_profile})
        df.to_csv(csv_name, index=False)
        print(f'file {csv_name} has been saved to disk')
    else:
        pass

def m_curv_plot_profile(data, m_curvature, position, normalized=True, save_file=False):

    """ Used for mirror radious scans, this function plots the vertical profile,
	it can save a CSV file with the radious step on the filename """
    
    try:
        label = f'Curvature scan: vertical profile for {m_curvature[position]} km'
        csv_name = f'Curvature_scan_ver_prof_{m_curvature[position]}_km_exp.csv'
    except:
        print(f'Radius of curvature step {m_curvature[position]} km has not been found')

    # values for the plotting #    
    exposure_time = 1 # 0 = 1 ms, 1 = 3 ms and 3 = 10 ms #
    f_size = 12
    #pixel_size = 3.75/2 #pixel size in um
    pixel_size = 2.87
    #scale = 2.4
    #origin_y = 600

    height, width = data[position, exposure_time, :].shape
    
    #y = np.linspace(0 - origin_y, height - origin_y, num=height) * scale

    y = np.linspace(-height/2, height/2, height) * pixel_size   

    v_profile = np.zeros_like(data[position, exposure_time, :, 0])	

    for i in range(width):
        v_profile = np.add(v_profile, data[position, exposure_time, :, i])

    if normalized:
        v_profile /= np.max(v_profile)
        plt.ylabel("Normilized intensity [a.u.]", fontsize= f_size)
    else:
        plt.ylabel("Intensity [a.u.]", fontsize= f_size)
    
    plt.plot(y, v_profile, label=label, linewidth=2)
    plt.xlabel("Vertical [$\mu$m]", fontsize= f_size)
        
    plt.xticks(fontsize= f_size)
    plt.yticks(fontsize= f_size)

    plt.legend(fontsize= f_size)

    plt.show()

    if save_file:
        df = pd.DataFrame({'vertical':y,'intensity':v_profile})
        df.to_csv(csv_name, index=False)
        print(f'file {csv_name} has been saved to disk')
    else:
        pass 

#### The following functions were used to the slit vertical scans  ###

def peak_position_fit(data, slit_positions, plotting=True, save_files=True):

    """ This function was applied on the slit scan in order to get the peak
	position using a Gaussian fit, it can plot each slit scan step and save a 
	file with the peak position in function of the slit position"""    

    height, width = data[0, :, :].shape

    #y = np.linspace(0, height, height) #* pixel_size
    y = range(height)

    peak_intensities = []
    peak_positions = []

    # new range of integration over the vertical, to avoid noise data?
    #int_interval = np.arange(1042, 1053)

    for i in range(len(slit_positions)):
        v_profile = np.zeros_like(data[i, :, 0])

        #for element in int_interval:
        for element in range(width):        
            v_profile = np.add(v_profile, data[i, :, element])

        #define a range around the peak:
        peak_ind = np.where(v_profile == np.amax(v_profile))
        #print("peak index: ", peak_ind[0][0])        
        new_x = y[peak_ind[0][0] - 11 : peak_ind[0][0] + 11]
        new_y = v_profile[peak_ind[0][0] - 11 : peak_ind[0][0] + 11]

        #Fitting process for each slit position
        fit = FitManager()
        fit.setdata(x=new_x, y=new_y)
        fit.loadtheories(fittheories)
        fit.settheory('Gaussians') #'Gaussians', 'Lorentz'
        fit.setbackground('Linear')
        fit.estimate()
        fit.runfit()

        b, m, height1, pp, fwhm = (param['fitresult'] for param in fit.fit_results)

        peak_pos = np.round(pp, 3)

        print("peak position from Gaussian: ,", peak_pos)               

        if plotting:
            #Atenttion, this will plot for every slit position
            new_y -= m * new_x + b            
            plot(new_x, new_y, new_x, sum_gauss(np.array(new_x), *[height1, pp, fwhm]), legend=["data", "gaussian fit"])            
            
        peak_positions.append(peak_pos)
        #area = np.round(fit.fit_results[2]["fitresult"] * fit.fit_results[4]["fitresult"] * 0.5 * np.pi, 2) #Lorentz
        a_gauss = np.sqrt(2*np.pi) * height1 * fwhm / (2 * np.sqrt(2 * np.log(2))) # gauss
        peak_intensities.append(a_gauss)
        print("Area of gaussian: ", a_gauss)
    
    if save_files:
        df_intens = pd.DataFrame({'vertical':slit_positions,'intensity':peak_intensities})
        df_intens.to_csv('gauss_v_profile_slit_pos.csv', index=False)
        print('file gauss_v_profile_slit_pos.csv has been saved to disk')

        df_pos = pd.DataFrame({'Slit_pos [mm]':slit_positions,'peak_position[pixels]':peak_positions})
        df_pos.to_csv('gauss_slit_pos_vs_peak_pos.csv', index=False)
        print('file gauss_slit_pos_vs_peak_pos.csv has been saved to disk')

    else:
        print("any file has been saved")

    return np.array(peak_positions), np.array(peak_intensities)

    
def add_images(data, slit_positions, plot='image', save_file = 'image'):

    """ This function adds all the 2D images for each stlit scan step and then
	extracts the vertical projection"""

    sum_img = np.zeros_like(data[0, :, :])

    for i, step in enumerate(slit_positions):

        sum_img = sum_img + data[i, :, :]

    height, width = sum_img.shape

    v_pixels = np.linspace(0, height, height) #* pixel_size    

    v_profile = np.zeros_like(sum_img[:, 0])	

    #for i in range(width):
    #for i in np.arange(1037, 1058, 1):
    for i in np.arange(1045, 1051, 1):
        v_profile = np.add(v_profile, sum_img[:, i])

    if plot=='image':

        height, width = sum_img.shape

        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)	
    
        plt.pcolormesh(x, y, sum_img, cmap=plt.cm.viridis, shading='auto')               

    elif plot == 'profile':

        plt.plot(v_pixels, v_profile)

    if save_file == 'image':

        hf = h5py.File('sum_image.h5', 'w')

        hf.create_dataset('sum_img', data=sum_img)

        hf.close()

    elif save_file == 'profile':
        df = pd.DataFrame({'vertical':v_pixels,'intensity':v_profile})
        df.to_csv('v_profile_pixels_selected.csv', index=False)
        print('file v_profile_pixels_selected.csv has been saved to disk')   
    
    return sum_img, v_pixels, v_profile
    
if __name__=="__main__":
    pass
    
    #### examples of use ####
    
    #data_curvature, m_curvature = get_data('xeye004.h5')
    #m_curv_plot_profile(data_curvature, m_curvature, 10, normalized=True, save_file=False)	

    #data, slit_positions = get_data('xeye006.h5')
#
    #peaks = []
#
    #for pos in slit_positions:
#
    #    peak = peak_position(data, slit_positions, pos, plot=True)
#
    #    peaks.append(peak)




    #plot_profile('xeye003.h5', 50, normalized=True, save_file=False)
    #data, m_curvature = get_data('xeye004.h5')
