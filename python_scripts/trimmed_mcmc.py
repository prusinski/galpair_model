import numpy as np
from astropy import units as u, constants as c
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import matplotlib
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.optimize import fsolve, fmin
from astropy.io import fits
import scipy.interpolate as interp
import emcee
import corner
from multiprocessing import Pool
import multiprocessing as mp

cnfw = 3.3 # NFW concentration parameter
z = 2.3 # redshift
Rs = 27 # kpc = Rvir = 90 kpc
H = cosmo.H(z).to('km/s/kpc').value

noise_level = 0.02
A = 1.2e7 #* u.km**2/u.s**2

bres = 5 # kpc
min_b = 10 # kpc
max_b = 250 # kpc
bvec = np.arange(min_b, max_b, bres)
vrawsamp = np.arange(-1000, 1015, 30)

bm, vm = np.meshgrid(bvec, vrawsamp)

bvec_final = np.arange(min_b, max_b+2.5, 5)
vvec_final = np.arange(-1000, 1025, 50)

lya_real = fits.open('../lya_conv_250311.fits')[0].data
lya_err_real = fits.open('../lya_error_conv_250311.fits')[0].data

def vins(r, voff):
    vin = voff + H*r
    return vin

def vouts(r, vi):
    vout = np.sqrt(vi**2 + A*(-np.log((Rs+1)/Rs) + (1/r * np.log((Rs+r)/Rs))))
    return vout

def ains(r, a0in, gin):
    ain = (a0in * (r/100)**(-gin))
    return ain

def aouts(r, a0out, gout):
    aout = (a0out * (r/100)**(-gout))
    return aout



def model(theta):
    vi, voff, a0out, a0in, gout, gin = theta

    rmax = 500 #kpc
    rmax_out = rmax
    rmax_in = rmax
    # rmax_out = 100 * (a0out/noise_level)**(1/gout)
    # rmax_out = np.min([rmax_out_est, rmax])
    # print(rmax_out)
    # rmax_in_est = 100 * (a0in/noise_level)**(1/gin)

    # rmax_in = np.min([rmax_in_est, rmax])
    # print(rmax_in)


    rawhmap = np.zeros((len(vrawsamp), len(bvec)))



    for bi, b in enumerate(bvec):

        lmax_out = np.sqrt(rmax_out**2 - b**2)
        lmax_in = np.sqrt(rmax_in**2 - b**2) #np.nanmin([np.sqrt((-voff/H)**2-b**2), np.sqrt(rmax_in**2 - b**2)])


        # larr_slowl_out = np.linspace(-lmax_out, -200, 50)
        # nfast_out = int(vi/900 * 1200 * np.exp(-b/200)) #1500
        # larr_fast_out = np.linspace(-200, 200, nfast_out)
        # larr_slowr_out = np.linspace(200, lmax_out, 50)
        # larr_out = np.concatenate([larr_slowl_out, larr_fast_out, larr_slowr_out])

        larr_out = np.linspace(-lmax_out, lmax_out, 2000)

        r_out = np.sqrt(larr_out**2 + b**2)

        tau_outs = aouts(r_out, a0out, gout)
        tau_outs_arr = np.array(tau_outs)
        # print(tau_outs_arr)


        # larr_slowl_in = np.linspace(-lmax_in, -100, 50)
        # nfast_in = int(500 * np.exp(-b/100)) #1500
        # larr_fast_in = np.linspace(-100, 100, nfast_in)
        # larr_slowr_in = np.linspace(100, lmax_in, 50)
        # larr_in = np.concatenate([larr_slowl_in, larr_fast_in, larr_slowr_in])

        larr_in = np.linspace(-lmax_in, lmax_in, 2000)
        r_in = np.sqrt(larr_in**2 + b**2)

        # lmaxcalc = np.sqrt((-voff/H)**2-b**2)
        # print(lmaxcalc)

        tau_ins = ains(r_in, a0in, gin)
        tau_ins_arr = np.array(tau_ins)

        # plt.figure()
        # plt.plot(larr_out, tau_outs, 'b.')
        # # plt.plot(larr_in, tau_ins, 'r.')
        # plt.show()


        vLOS_out = larr_out/r_out*vouts(r_out, vi)
        vLOS_in = larr_in/r_in*vins(r_in, voff)
        # plt.figure()
        # sc = plt.scatter(larr_out, vLOS_out, c=tau_outs)
        # sc1 = plt.scatter(larr_in, vLOS_in, c=tau_ins, cmap='inferno')
        # plt.xlabel('$\\ell$ (kpc)')
        # plt.ylabel('$v_\\mathrm{LOS}$ (km/s)')
        # plt.colorbar(sc, label = '$\\tau_\\mathrm{out}$')
        # plt.colorbar(sc1, label = '$\\tau_\\mathrm{in}$')
        # plt.title(f'$b={b}$ kpc')
        # plt.axhline(0, c = 'k', alpha = 0.7, lw = 0.7, ls = '--')
        # plt.axvline(0, c = 'k', alpha = 0.7, lw = 0.7, ls = '--')
        # plt.tight_layout()
        # plt.show()

        maxvout = np.nanmax(vLOS_out)
        maxvin = np.nanmax(vLOS_in)

        # bad b's
        if ~np.isfinite(maxvout):
            for j in vrawsamp:
                taulist_out.append([j, 0])
            continue

        if ~np.isfinite(maxvin):
            # print(f'bad {b}')
            for j in vrawsamp:
                taulist_in.append([j, 0])
            continue

        l_minvout = larr_out[np.argmin(vLOS_out)]
        l_maxvout = larr_out[np.argmax(vLOS_out)]

        l_minvin = larr_in[np.argmin(vLOS_in)]
        l_maxvin = larr_in[np.argmax(vLOS_in)]


        taulist_out = []
        taulist_in = []

        for vl in vrawsamp:
            # outflow
            if np.abs(vl) > maxvout:
                taulist_out.append([vl+15, 0])
            else:
                possible_values = larr_out[(vLOS_out > vl) & (vLOS_out < vl+30)]
                l_near = possible_values[(possible_values > l_minvout) & (possible_values < l_maxvout)]
                l_far = possible_values[(possible_values < l_minvout) | (possible_values > l_maxvout)]

                tau_near_inds = [np.argwhere(larr_out == ln)[0][0] for ln in l_near]
                tau_out_near = tau_outs_arr[tau_near_inds]


                tau_far_inds = [np.argwhere(larr_out == ln)[0][0] for ln in l_far]
                tau_out_far = tau_outs_arr[tau_far_inds]
                tau_tot_out = np.nan_to_num(np.nanmean(tau_out_near)) + np.nan_to_num(np.nanmean(tau_out_far))

                if np.isfinite(tau_tot_out):
                    taulist_out.append([vl+15, tau_tot_out])

            # inflow
            if np.abs(vl) > maxvin:
                taulist_in.append([vl+15, 0])
            else:
                possible_values = larr_in[(vLOS_in > vl) & (vLOS_in < vl+30)]
                # print(possible_values)
                l_near = possible_values[(possible_values < l_minvin) & (possible_values > l_maxvin)]
                l_far = possible_values[(possible_values > l_minvin) | (possible_values < l_maxvin)]

                tau_near_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_near]
                tau_in_near = tau_ins_arr[tau_near_inds]


                tau_far_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_far]
                tau_in_far = tau_ins_arr[tau_far_inds]
                tau_tot_in = np.nan_to_num(np.nanmean(tau_in_near)) + np.nan_to_num(np.nanmean(tau_in_far))
                # print(tau_tot)

                if np.isfinite(tau_tot_in):
                    taulist_in.append([vl+15, tau_tot_in])

        tauarr_out_wzeros = np.array(taulist_out) # v, tau
        tauarr_in_wzeros = np.array(taulist_in)

        tauarr_out = tauarr_out_wzeros[tauarr_out_wzeros[:,1] > 0]
        tauarr_in = tauarr_in_wzeros[tauarr_in_wzeros[:,1] > 0]

        if len(tauarr_out) == 0:
            tout = vrawsamp*0.0
        else:
            tout = np.interp(vrawsamp, tauarr_out[:,0], tauarr_out[:,1], left=0, right=0)

        if len(tauarr_in) == 0:
            tin = vrawsamp*0.0
        else:
            tin = np.interp(vrawsamp, tauarr_in[:,0], tauarr_in[:,1], left=0, right=0)

        rawhmap[:,bi] = tout + tin
            # tau_tot = tau_tot_out + tau_tot_in
            # taulist.append([b, vl+15, tau_tot])




    # points = tauarr[:,0:2]
    # values = tauarr[:,2]
    # rawhmap = interp.griddata(points, values, (bm, vm), method='nearest')

    # plt.figure()
    # plt.imshow(rawhmap, origin='lower')
    # plt.show()




    # print(tauarr_out[np.isfinite(tauarr_out[:,2])])
    # plt.figure()
    # plt.scatter(tauarr_out[:,1], tauarr_out[:,2])
    # plt.plot(vrawsamp, tout)
    # plt.plot(vrawsamp, tin)
    # plt.show()

    # plt.figure()
    # sc = plt.scatter(tauarr_out[:,0], tauarr_out[:,1], c=tauarr_out[:,2]) #/cmap='inferno_r' vmin=2e-2, vmax = 1e-1 norm=matplotlib.colors.LogNorm()
    # plt.colorbar(sc, label = '$\\tau_\\mathrm{out}$')

    # # tin = plt.scatter(tauarr_in[:,0], tauarr_in[:,1], c=tauarr_in[:,2], cmap='plasma_r') #/cmap='inferno_r' vmin=2e-2, vmax = 1e-1 norm=matplotlib.colors.LogNorm()
    # # plt.colorbar(tin, label = '$\\tau_\\mathrm{in}$')
    # plt.xlabel('b (kpc)')
    # plt.ylabel('$v_\\mathrm{LOS}$ (km/s)')
    # plt.tight_layout()
    # plt.show()

    # make map

    # vres = 10 #50
    # vlosr = np.arange(-1000, 1000, vres)

    # hmap = np.zeros((len(vlosr), len(bvec)))

    # for bi, b in enumerate(bvec):
    #     bcut_out = (tauarr_out[:,0] == b)
    #     bcut_in = (tauarr_in[:,0] == b)

    #     for vind, v in enumerate(vlosr):
    #         vcut_out = (tauarr_out[:,1] > v) & (tauarr_out[:,1] < v+vres)
    #         vcut_in = (tauarr_in[:,1] > v) & (tauarr_in[:,1] < v+vres)

    #         tauavg_out = np.nanmean(tauarr_out[:,2][bcut_out & vcut_out])
    #         if ~np.isfinite(tauavg_out):
    #             tauavg_out = 0

    #         tauavg_in = np.nanmean(tauarr_in[:,2][bcut_in & vcut_in])
    #         if ~np.isfinite(tauavg_in):
    #             tauavg_in = 0

    #         hmap[vind,bi] = tauavg_out + tauavg_in

    hmap_conv = convolve(rawhmap, Gaussian2DKernel(2,2), boundary = 'extend') # 100 km/s and 5 kpc sampling

    f = interp.RectBivariateSpline(vrawsamp, bvec, hmap_conv, kx=3, ky=3)
    hmap_reshaped = f(vvec_final, bvec_final)

    # print(new_im.shape)

    # plt.figure()
    # plt.imshow(hmap_reshaped, aspect='auto', origin='lower', extent = [bvec_final.min(), bvec_final.max(), vvec_final.min(), vvec_final.max()])
    # plt.show()

    # # # print(hmap.shape)
    # plt.figure()
    # us = plt.imshow(rawhmap, origin='lower', extent = (bvec[0], bvec[-1], vrawsamp[0], vrawsamp[-1]),
    #            aspect='auto', cmap='plasma')
    # plt.colorbar(us, label = '$\\tau_\\mathrm{tot}$')
    # plt.xlabel('b (kpc)')
    # plt.ylabel('$v_\\mathrm{LOS}$ (km/s)')
    # plt.title('Unsmoothed')
    # # plt.xscale('log')
    # plt.show()

    # plt.figure()
    # sm = plt.imshow(hmap_conv, origin='lower', extent = (bvec[0], bvec[-1], vrawsamp[0], vrawsamp[-1]),
    #            aspect='auto', cmap='inferno')
    # plt.colorbar(sm, label = '$\\tau_\\mathrm{tot}$')
    # plt.xlabel('b (kpc)')
    # plt.ylabel('$v_\\mathrm{LOS}$ (km/s)')
    # plt.title('Smoothed')
    # # plt.xscale('log')
    # plt.show()

    return hmap_reshaped




def lnlike(theta):
    # return -0.5 * np.nansum(((lya_real - model(theta))/lya_err_real)**2)
    return -0.5 * np.nansum((lya_real - model(theta))**2)

def lnprior(theta):
    vi, voff, a0out, a0in, gout, gin = theta

    vir = (400 < vi < 900)
    voffr = (-600 < voff < 0)
    a0outr = (0 < a0out < 0.4)
    a0inr = (0. < a0in < 0.4)
    goutr = (0 < gout < 1.5)
    ginr = (0. < gin < 1.5)

    if vir and voffr and a0outr and a0inr and goutr and ginr:
        return 0.0

    return -np.inf

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

if __name__ == '__main__':

    nwalkers = 60
    niter = 7500
    initial = np.array([750, -250, 0.28, 0.1, 0.7, 0.37])
    ndims = len(initial)

    p0 = [initial * (1 + 0.01*np.random.randn(ndims)) for i in range(nwalkers)]
    # print(p0)

    filename = "../MCMC_outputs/trimmed_60w_7500it_250311-both.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndims)

    with Pool(10) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, pool=pool, backend=backend)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        # return sampler, pos, prob, state
