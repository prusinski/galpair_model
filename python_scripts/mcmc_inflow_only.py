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
A = 1.2e7 #* u.km**2/u.s**2

bres = 2 # kpc
min_b = 100 # kpc
max_b = 250 # kpc
bvec = np.arange(min_b, max_b, bres)
vrawsamp = np.arange(-1000, 1000, 30)

bvec_final = np.arange(min_b, max_b+2.5, 5)
vvec_final = np.arange(-1000, 1050, 100)

lya_real_full = fits.open('lya_conv_240618.fits')[0].data
lya_b_vec = np.arange(10,252.5,5)
lya_v_vec = np.arange(-1000, 1050, 100)
lya_real = lya_real_full[:, lya_b_vec >= 100]
lya_real.shape


def model(theta):
    # vi, voff, a0out, a0in, gout, gin = theta
    voff, a0in, gin = theta

    # for the moment, no outflows
    vi = 638
    a0out = 0 #0.29
    gout = 0.21

    def vins(r):
        vin = voff + H*r
        return vin

    def vouts(r):
        vout = np.sqrt(vi**2 + A*(-np.log((Rs+1)/Rs) + (1/r * np.log((Rs+r)/Rs))))
        return vout


    rmax_out = fmin(vouts, 35, ftol=0.5, disp=False)[0]
    # print(rmax_out)

    rmax_in = fsolve(vins, 100)[0] #xtol = 1
    # print(rmax_in)

    def ains(r):
        ain = (a0in * (r/100)**(-gin))
        return ain

    def aouts(r):
        if r > rmax_out:
            aout=0
        else:
            aout = (a0out * (r/100)**(-gout))
        return aout


    taulist_out = []
    taulist_in = []


    for bi, b in enumerate(bvec):

        lmax_out = np.sqrt(rmax_out**2 - b**2)
        lmax_in = np.sqrt(rmax_in**2 - b**2)

        larr_slowl_out = np.linspace(-lmax_out, -50, 50)
        nfast_out = int(150 * np.exp(-b/100)) #1500
        larr_fast_out = np.linspace(-50, 50, nfast_out)
        larr_slowr_out = np.linspace(50, lmax_out, 50)
        larr_out = np.concatenate([larr_slowl_out, larr_fast_out, larr_slowr_out])

        tau_outs = [aouts(np.sqrt(b**2 + li**2)) for li in larr_out]


        larr_slowl_in = np.linspace(-lmax_in, -50, 10)
        nfast_in = int(100 * np.exp(-b/100)) #1500
        larr_fast_in = np.linspace(-50, 50, nfast_in)
        larr_slowr_in = np.linspace(50, lmax_in, 10)
        larr_in = np.concatenate([larr_slowl_in, larr_fast_in, larr_slowr_in])
        tau_ins = [ains(np.sqrt(b**2 + li**2)) for li in larr_in]
        # print(tau_ins)


        r_out = np.sqrt(larr_out**2 + b**2)
        r_in = np.sqrt(larr_in**2 + b**2)
        # vout, vin = vels(r*u.kpc)

        vLOS_out = larr_out/r_out*vouts(r_out)
        vLOS_in = larr_in/r_in*vins(r_in)
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



        l_minvout = larr_out[np.argmin(vLOS_out)]
        l_maxvout = larr_out[np.argmax(vLOS_out)]
        maxvout = np.nanmax(vLOS_out)
        if ~np.isfinite(maxvout):
            # print(f'bad {b}')
            for j in vrawsamp:
                taulist_out.append([b, j, 0])
        else:
            # outflow
            for i in range(int(np.floor(-maxvout)), int(np.ceil(maxvout)), 30): # 10 km/s

                possible_values = larr_out[(vLOS_out > i) & (vLOS_out < i+30)]
                l_near = possible_values[(possible_values > l_minvout) & (possible_values < l_maxvout)]
                l_far = possible_values[(possible_values < l_minvout) | (possible_values > l_maxvout)]

                tau_near_inds = [np.argwhere(larr_out == ln)[0][0] for ln in l_near]
                tau_out_near = np.array(tau_outs)[tau_near_inds]


                tau_far_inds = [np.argwhere(larr_out == ln)[0][0] for ln in l_far]
                tau_out_far = np.array(tau_outs)[tau_far_inds]
                tau_tot = np.nanmean(tau_out_near) + np.nanmean(tau_out_far)

                if np.isfinite(tau_tot):
                    taulist_out.append([b, i+15, tau_tot])

        #inflow

        l_minvin = larr_in[np.argmin(vLOS_in)]
        l_maxvin = larr_in[np.argmax(vLOS_in)]
        maxvin = np.nanmax(vLOS_in)
        # print(l_minvin, l_maxvin, maxvin)

        for i in range(int(np.floor(-maxvin)), int(np.ceil(maxvin)), 30):

            possible_values = larr_in[(vLOS_in > i) & (vLOS_in < i+30)]
            # print(possible_values)
            l_near = possible_values[(possible_values < l_minvin) & (possible_values > l_maxvin)]
            l_far = possible_values[(possible_values > l_minvin) | (possible_values < l_maxvin)]

            tau_near_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_near]
            tau_in_near = np.array(tau_ins)[tau_near_inds]


            tau_far_inds = [np.argwhere(larr_in == ln)[0][0] for ln in l_far]
            tau_in_far = np.array(tau_ins)[tau_far_inds]
            tau_tot = np.nanmean(tau_in_near) + np.nanmean(tau_in_far)
            # print(tau_tot)

            if np.isfinite(tau_tot):
                taulist_in.append([b, i+15, tau_tot])


    tauarr_out = np.array(taulist_out)
    tauarr_in = np.array(taulist_in)


    rawhmap = np.zeros((len(vrawsamp), len(bvec)))

    if len(tauarr_in) == 0:
        rawhmap[:,:] = 0.
    else:
        for bi, b in enumerate(bvec):
            bcut_out = (tauarr_out[:,0] == b)
            bcut_in = (tauarr_in[:,0] == b)
            if len(tauarr_out[bcut_out]) == 0:
                tout = vrawsamp*0.0
            else:
                tout = np.interp(vrawsamp, tauarr_out[:,1][bcut_out], tauarr_out[:,2][bcut_out], left=0, right=0)

            if len(tauarr_in[bcut_in]) == 0:
                tin = vrawsamp*0.0
            else:
                tin = np.interp(vrawsamp, tauarr_in[:,1][bcut_in], tauarr_in[:,2][bcut_in], left=0, right=0)

            rawhmap[:,bi] = tout + tin

    hmap_conv = convolve(rawhmap, Gaussian2DKernel(2,1), boundary = 'extend') # 100 km/s and 5 kpc sampling

    f = interp.RectBivariateSpline(vrawsamp, bvec, hmap_conv, kx=3, ky=3)
    hmap_reshaped = f(vvec_final, bvec_final)


    return hmap_reshaped

def lnlike(theta, truth = lya_real):
    return -0.5 * np.nansum((lya_real - model(theta))**2)

def lnprior(theta):
    # vi, voff, a0out, a0in, gout, gin = theta
    voff, a0in, gin = theta

    # vir = (450 < vi < 700)
    voffr = (-1000 < voff < -20)
    # a0outr = (0 < a0out < 1)
    a0inr = (0 < a0in < 1)
    # goutr = (0 < gout < 0.5)
    ginr = (0 < gin < 1)

    if voffr and a0inr and ginr:
        return 0.0

    return -np.inf

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

if __name__ == '__main__':

    nwalkers = 20
    niter = 4000
    initial = np.array([-144, 0.83, 0.1])
    ndims = len(initial)

    p0 = initial_pos = [initial * (1 + 0.01*np.random.randn(ndims)) for i in range(nwalkers)]

    filename = "inflow_20w_4000it.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndims)

    with Pool(10) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, pool=pool, backend=backend)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        # return sampler, pos, prob, state

    # fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # labels = ["vmax", "alpha_out", "gout"]
    # for i in range(ndims):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)
    #
    # axes[-1].set_xlabel("step number");
    # plt.show()
    #
    # tau = sampler.get_autocorr_time()
    # print(tau)
    #
    # flat_samples = sampler.get_chain(discard=25, flat=True) # discard=100, thin=15, flat=True
    # print(flat_samples.shape)
    #
    # fig = corner.corner(
    #     flat_samples, labels=labels, truths=initial
    # );
    # plt.show()
