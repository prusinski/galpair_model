# Galaxy Pair Project (10/2023 - 7/2024)

Constructing a idealized galaxy model with outflow and inflow components based on Chen+20 and Cameron's `idealized_galaxy` repository. 

## Order of operations

- `chen+20_model` is the preliminary model and testing space
- `chen+20_model_slow` has units and is slow as a result
- `chen+20_model_fast` gets rid of units and is considerably faster
- `chen+20_MCMC` sets up a parallelized MCMC but only works on `mizar` in a jupyter environment. Can't run parallel process in jupyter environment on a Mac.
- `chen+20_MCMC_check` has the best fit (outflow only) and lets you play with each parameter individually
- `constant_inflow` has a constant inflow as the name suggests
- `out+in` was a test to see if you combined the outflow and inflow as one component what the profile would look like. Looks kind of weird but maybe consistent with the two component model. Not going to pursue.
- `no_abs_radial_dependence` was supposed to have a constant absorption coefficient but I don't think this worked, or I didn't get that far.
- `chen+20_adjust_gal_params` adjusted the parameters of the NFW halo - realized a different box configuration was required since the integration boxes became huge
- `chen+20_trimmed_box` only "integrates" down to a certain $\Delta\tau$ - anything else is below the noise level. This allows faster outflows e.g. 800 km/s. 

6/26/24: going to try running an MCMC using the trimmed box.

---

- `lya_centroid` looks at folding the Ly$\alpha$ profile and assuming some sort of symmetry.


`vbt_plots` makes `lya_map_<date>.fits` files and these can be convolved using `lya_map_convolve` to make `lay_conv_<date>.fits` files.


