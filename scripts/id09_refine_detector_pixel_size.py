# -*- coding: utf-8 -*-

import numpy
from srxraylib.plot.gol import plot, plot_image
from silx.math.fit.functions import sum_gauss
from silx.math.fit import fittheories
from silx.math.fit.fitmanager import FitManager

""" This set of functions was used to get a refiment of the xeye pixel size """

def get_fwhm(x,h):
    tt = numpy.where(h>=max(h)*0.5)
    if h[tt].size > 1:
        binSize = x[1]-x[0]
        fwhm = binSize*(tt[0][-1]-tt[0][0])
    else:
        fwhm = None

    return fwhm

def fit_gaussian(x,z1,do_plot=True):

    fwhm = get_fwhm(x,z1)
    print(" Graph FWHM is: ",fwhm)
    p = [z1.max(),0,fwhm]

    fit = FitManager()
    fit.setdata(x=x, y=z1)
    fit.loadtheories(fittheories)
    fit.settheory('Gaussians')
    fit.estimate()
    fit.runfit()

    print("Searched parameters = %s" % p)
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.fit_results:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)
    fwhm_txt = "FWHM of fit = %5.3f um"%(fit.fit_results[2]['fitresult'])

    z11 = sum_gauss(x, *dummy_list)

    if do_plot:
        plot(x,z1,x,z11,legend=["data","fit"],ylog=False)


    Height = fit.fit_results[0]['fitresult']
    Position = fit.fit_results[1]['fitresult']
    FWHM = fit.fit_results[2]['fitresult']

    return Height, Position, FWHM, z11



def pixels_to_mm(pixel_array, pixel_size, pixel_shift=0.0, linearity_correction=0):
    p1 = -(pixel_array + pixel_shift)
    return  p1 * (pixel_size + linearity_correction * p1)

def get_y(y, x=None, use_derivative=False):
    if use_derivative:
        return numpy.gradient(y / y.max(), x)
    else:
        return y / y.max()


if __name__ == "__main__":

    do_plot = 0

    aP = numpy.loadtxt("v_profile_vs_pixels.csv", delimiter=',', skiprows=1)
    aS = numpy.loadtxt("v_profile_vs_slit_pos.csv", delimiter=',', skiprows=1)



    pixel_size = 2.8e-3
    pixel_shift = -615


    xS = aS[:,0]
    yS = get_y(aS[:,1])
    # yS = get_y(aS[:,1], xS, use_derivative=True)

    xP = pixels_to_mm(aP[:,0], pixel_size, pixel_shift=pixel_shift)
    yP = get_y(aP[:,1])
    # yP = get_y(aP[:,1], xP, use_derivative=True)


    # interpolate
    if xP[1] - xP[0] > 0:
        yPi = numpy.interp(xS, xP, yP)
    else:
        yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))


    mydiff = numpy.abs(yPi - yS)

    # Height, Position, FWHM, z11 = fit_gaussian(xS, mydiff, do_plot=0)

    if 1:
        z111 = sum_gauss(xS, 1.0, 0.10, 0.9* 10)
        z111 += yP.min()
        i0 = yPi.size // 2
        amplitude_correction = z111 * (yS[i0] / yPi[i0] / z111[i0]) # (z111 / z111[numpy.argmax(z111)])
    else:
        amplitude_correction = z11 * 0.0 + 1

    if do_plot: plot(
        xP, yP,
        xS, yS,
        xS, yPi,
        xS, mydiff,
        xS, z111,
        xS, yPi * amplitude_correction,
        legend=["vs pixel", "vs slit position", "P interpolated", "diff", "diff fit", "vs pixel amplitude corrected"],
        )


    for kk in range(25):
        # 1D optimization scan pixel_size
        if 1:
            #
            pixel_values = numpy.linspace(2.6e-3, 3.0e-3, 100)
            diff_values = numpy.zeros_like(pixel_values)
            for i in range(pixel_values.size):
                pixel_size = pixel_values[i]
                xP = pixels_to_mm(aP[:, 0], pixel_size, pixel_shift=pixel_shift)
                yP = get_y(aP[:, 1], xP, use_derivative=True)
                yS = get_y(aS[:, 1], xS, use_derivative=True)
                # interpolate
                if xP[1] - xP[0] > 0:
                    yPi = numpy.interp(xS, xP, yP)
                else:
                    yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
                diff_values[i] = numpy.abs(yPi - yS).sum()

            i0 = numpy.argmin(diff_values)
            pixel_size = pixel_values[i0]
            print("Best pixel size found: ",  pixel_size * 1e3)

            if do_plot: plot(pixel_values * 1e3, diff_values)

            xP = pixels_to_mm(aP[:,0], pixel_size, pixel_shift=pixel_shift)
            yS = get_y(aS[:, 1])
            yP = get_y(aP[:,1])
            # interpolate
            if xP[1] - xP[0] > 0:
                yPi = numpy.interp(xS, xP, yP)
            else:
                yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
            # yP = get_y(aP[:,1], xP, use_derivative=True)

            if do_plot: plot(
                xP, yP,
                xS, yS,
                xS, yPi * amplitude_correction,
                legend=["vs pixel", "vs slit position", "vs pixel corrected amplitude"],
                title="After best pixel size",
                )

        # 1D optimization scan shift value
        if 1:
            #
            shift_values = numpy.linspace(-620, -610, 100)
            diff_values = numpy.zeros_like(shift_values)
            for i in range(shift_values.size):
                xP = pixels_to_mm(aP[:, 0], pixel_size, pixel_shift=shift_values[i])
                yP = get_y(aP[:, 1], xP, use_derivative=True)
                yS = get_y(aS[:, 1], xS, use_derivative=True)
                # interpolate
                if xP[1] - xP[0] > 0:
                    yPi = numpy.interp(xS, xP, yP)
                else:
                    yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
                diff_values[i] = numpy.abs(yPi - yS).sum()

            i0 = numpy.argmin(diff_values)
            pixel_shift = shift_values[i0]
            print("Best pixel shift found: ",  pixel_shift)

            if do_plot: plot(shift_values, diff_values)

            xP = pixels_to_mm(aP[:,0], pixel_size, pixel_shift=pixel_shift)
            yS = get_y(aS[:, 1])
            yP = get_y(aP[:,1])
            # interpolate
            if xP[1] - xP[0] > 0:
                yPi = numpy.interp(xS, xP, yP)
            else:
                yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
            # yP = get_y(aP[:,1], xP, use_derivative=True)

            if do_plot: plot(
                xP, yP,
                xS, yS,
                xS, yPi * amplitude_correction,
                legend=["vs pixel", "vs slit position", "vs pixel corrected amplitude"],
                title="After best shift",
                )


        # 1D optimization REFINE NON LINEAR TERM
        nonlin_best = 0.0
        if 1:
            # scan pixel_size
            nonlin_values = numpy.linspace(-2.25e-6, 3.5e-6, 100)
            diff_values = numpy.zeros_like(nonlin_values)
            for i in range(nonlin_values.size):
                xP = pixels_to_mm(aP[:, 0], pixel_size, pixel_shift=pixel_shift, linearity_correction=nonlin_values[i])
                yP = get_y(aP[:, 1], xP, use_derivative=True)
                yS = get_y(aS[:, 1], xS, use_derivative=True)
                # interpolate
                if xP[1] - xP[0] > 0:
                    yPi = numpy.interp(xS, xP, yP)
                else:
                    yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
                diff_values[i] = numpy.abs(yPi * amplitude_correction - yS).sum()

            i0 = numpy.argmin(diff_values)
            nonlin_best = nonlin_values[i0]
            print("Best nonlinearity: ",  nonlin_best)

            if do_plot: plot(nonlin_values, diff_values)

            xP = pixels_to_mm(aP[:,0], pixel_size, pixel_shift=pixel_shift, linearity_correction=nonlin_best)
            yS = get_y(aS[:, 1])
            yP = get_y(aP[:,1])

            # interpolate
            if xP[1] - xP[0] > 0:
                yPi = numpy.interp(xS, xP, yP)
            else:
                yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
            # yP = get_y(aP[:,1], xP, use_derivative=True)

            if do_plot: plot(
                xP, yP,
                xS, yS,
                xS, yPi * amplitude_correction,
                legend=["vs pixel", "vs slit position", "vs pixel corrected amplitude"],
                title="After best non-linear term",
                )


    import matplotlib.pylab as plt
    plot(
        xP, yP,
        xS, yS,
        xS, yPi * amplitude_correction,
        legend=["vs pixel", "vs slit position", "vs pixel corrected amplitude"],
        title="After %s iterations size: %f, shift: %d, nonlinearity: %g" % (kk+1, pixel_size*1e3, pixel_shift, nonlin_best),
        show=0
        )
    plt.grid()
    plt.show()


    # 2D optimization
    if 0:
        pixel_values = numpy.linspace(pixel_size * 0.9, pixel_size * 1.1, 100)
        shift_values = numpy.linspace(-620, -610, 100)
        diff_map = numpy.zeros((pixel_values.size, shift_values.size))

        for i in range(pixel_values.size):
            for j in range(shift_values.size):
                pixel_size = pixel_values[i]
                pixel_shift = shift_values[j]
                xP = pixels_to_mm(aP[:, 0], pixel_size, pixel_shift=pixel_shift)
                yP = get_y(aP[:, 1], xP, use_derivative=True)
                yS = get_y(aS[:, 1], xS, use_derivative=True)
                # interpolate
                if xP[1] - xP[0] > 0:
                    yPi = numpy.interp(xS, xP, yP)
                else:
                    yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))
                diff_map[i,j] = numpy.abs(yPi - yS).sum()



        minvalue = diff_map.min()
        for i in range(pixel_values.size):
            for j in range(shift_values.size):
                if diff_map[i,j] == minvalue:
                    i0 = i
                    j0 = j
        print(minvalue,i0,j0,pixel_values[i0] * 1e3, shift_values[j0])

        plot_image(diff_map, pixel_values * 1e3, shift_values, aspect='auto')

        pixel_size = pixel_values[i]
        pixel_shift = shift_values[j]

        xP = pixels_to_mm(aP[:,0], pixel_size, pixel_shift=pixel_shift)
        xS = aS[:,0]

        yP = get_y(aP[:,1])
        yS = get_y(aS[:,1])

        # yP = get_y(aP[:,1], xP, use_derivative=True)
        # yS = get_y(aS[:,1], xS, use_derivative=True)


        # interpolate
        if xP[1] - xP[0] > 0:
            yPi = numpy.interp(xS, xP, yP)
        else:
            yPi = numpy.interp(xS, numpy.flip(xP), numpy.flip(yP))

        plot(
            xP, yP,
            xS, yS,
            legend=["vs pixel", "vs slit position"],
            )

