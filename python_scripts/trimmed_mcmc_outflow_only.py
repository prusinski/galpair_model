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

lya_real = fits.open('../lya_conv_240701.fits')[0].data
lya_err_real = fits.open('../lya_error_conv_240701.fits')[0].data

# def vins(r, voff):
#     vin = voff + H*r
#     return vin

def vouts(r, vi):
    vout = np.sqrt(vi**2 + A*(-np.log((Rs+1)/Rs) + (1/r * np.log((Rs+r)/Rs))))
    return vout

# def ains(r, a0in, gin):
#     ain = (a0in * (r/100)**(-gin))
#     return ain

def aouts(r, a0out, gout):
    aout = (a0out * (r/100)**(-gout))
    return aout



def model(theta):
    vi, a0out, gout = theta

    rmax_out = 100 * (a0out/noise_level)**(1/gout)
    # print(rmax_out)
    # rmax_in_est = 100 * (a0in/noise_level)**(1/gin)
    # rmax_in_max = 10000
    # rmax_in = np.min([rmax_in_est, rmax_in_max])
    # print(rmax_in)


    rawhmap = np.zeros((len(vrawsamp), len(bvec)))



    for bi, b in enumerate(bvec):

        lmax_out = np.sqrt(rmax_out**2 - b**2)
        # lmax_in = np.sqrt(rmax_in**2 - b**2)


        larr_slowl_out = np.linspace(-lmax_out, -200, 50)
        nfast_out = int(vi/900 * 1200 * np.exp(-b/200)) #1500
        larr_fast_out = np.linspace(-200, 200, nfast_out)
        larr_slowr_out = np.linspace(200, lmax_out, 50)
        larr_out = np.concatenate([larr_slowl_out, larr_fast_out, larr_slowr_out])
        r_out = np.sqrt(larr_out**2 + b**2)

        tau_outs = aouts(r_out, a0out, gout)
        tau_outs_arr = np.array(tau_outs)


        # larr_slowl_in = np.linspace(-lmax_in, -100, 50)
        # nfast_in = int(500 * np.exp(-b/100)) #1500
        # larr_fast_in = np.linspace(-100, 100, nfast_in)
        # larr_slowr_in = np.linspace(100, lmax_in, 50)
        # larr_in = np.concatenate([larr_slowl_in, larr_fast_in, larr_slowr_in])
        # r_in = np.sqrt(larr_in**2 + b**2)
        #
        # tau_ins = ains(r_in, a0in, gin)
        # tau_ins_arr = np.array(tau_ins)


        vLOS_out = larr_out/r_out*vouts(r_out, vi)
        # vLOS_in = larr_in/r_in*vins(r_in, voff)
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
        # maxvin = np.nanmax(vLOS_in)

        # bad b's
        if ~np.isfinite(maxvout):
            for j in vrawsamp:
                taulist_out.append([j, 0])
            continue

        # if ~np.isfinite(maxvin):
        #     # print(f'bad {b}')
        #     for j in vrawsamp:
        #         taulist_in.append([j, 0])
        #     continue

        l_minvout = larr_out[np.argmin(vLOS_out)]
        l_maxvout = larr_out[np.argmax(vLOS_out)]

        # l_minvin = larr_in[np.argmin(vLOS_in)]
        # l_maxvin = larr_in[np.argmax(vLOS_in)]


        taulist_out = []
        # taulist_in = []

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
                tau_tot_out = np.nanmean(tau_out_near) + np.nan_to_num(np.nanmean(tau_out_far))

                if np.isfinite(tau_tot_out):
                    taulist_out.append([vl+15, tau_tot_out])

            # # inflow
            # if np.abs(vl) > maxvin:
            #     taulist_in.append([vl+15, 0])
            # else:
            #     possible_values = larr_in[(vLOS_in > vl) & (vLOS_in < vl+30)]
            #     # print(possible_values)
            #     l_near = possible_values[(possible_values < l_minvin) & (possible_values > l_maxvin)]
            #     l_far = possible_values[(possible_values > l_minvin) | (possible_values < l_maxvin)]
            #
            #     tau_near_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_near]
            #     tau_in_near = tau_ins_arr[tau_near_inds]
            #
            #
            #     tau_far_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_far]
            #     tau_in_far = tau_ins_arr[tau_far_inds]
            #     tau_tot_in = np.nan_to_num(np.nanmean(tau_in_near)) + np.nan_to_num(np.nanmean(tau_in_far))
            #     # print(tau_tot)
            #
            #     if np.isfinite(tau_tot_in):
            #         taulist_in.append([vl+15, tau_tot_in])

        tauarr_out = np.array(taulist_out) # v, tau
        # tauarr_in = np.array(taulist_in)

        if len(tauarr_out) == 0:
            tout = vrawsamp*0.0
        else:
            tout = np.interp(vrawsamp, tauarr_out[:,0], tauarr_out[:,1], left=0, right=0)

        # if len(tauarr_in) == 0:
        #     tin = vrawsamp*0.0
        # else:
        #     tin = np.interp(vrawsamp, tauarr_in[:,0], tauarr_in[:,1], left=0, right=0)

        rawhmap[:,bi] = tout # + tin
            # tau_tot = tau_tot_out + tau_tot_in
            # taulist.append([b, vl+15, tau_tot])

    hmap_conv = convolve(rawhmap, Gaussian2DKernel(1,1), boundary = 'extend') # 100 km/s and 5 kpc sampling

    f = interp.RectBivariateSpline(vrawsamp, bvec, hmap_conv, kx=3, ky=3)
    hmap_reshaped = f(vvec_final, bvec_final)

    return hmap_reshaped


def lnlike(theta):
    return -0.5 * np.nansum(((lya_real - model(theta))/lya_err_real)**2)

def lnprior(theta):
    vi, a0out, gout = theta

    vir = (450 < vi < 1000)
    # voffr = (-500 < voff < -20)
    a0outr = (0 < a0out < 1)
    # a0inr = (0 < a0in < 1)
    goutr = (0 < gout < 3)
    # ginr = (0 < gin < 3)

    if vir and a0outr and goutr:
        return 0.0

    return -np.inf

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

if __name__ == '__main__':

    nwalkers = 20
    niter = 1500
    initial = np.array([900, 0.07, 1])
    ndims = len(initial)

    p0 = [initial * (1 + 0.01*np.random.randn(ndims)) for i in range(nwalkers)]

    filename = "../MCMC_outputs/trimmed_outonly_20w_1500it.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndims)

    with Pool(7) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, pool=pool, backend=backend)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        # return sampler, pos, prob, state
