# -*- coding: utf-8 -*-
from pcraster._pcraster import *
from pcraster.framework import *

# import os
# import time
from copy import deepcopy

DEBUG_pest = False
global DEBUG_pest


def getConcAq(model, layer, mass, sorption_model="linear", gas=True):
    # Note that p_b (g/cm3) x k_d (L/Kg) -> unit-less
    theta_layer = model.theta[layer]
    depth = model.layer_depth[layer]
    if layer < 2:
        p_b = model.p_bAgr
    else:
        p_b = model.p_bZ

    if mapminimum(model.theta[layer]) < scalar(0):
        print("Moisture is " + float(mapminimum(model.theta[layer])) + " on layer: " + str(layer))

    if sorption_model == "linear":
        # Retardation factor (dimensionless)
        retard_layer = 1 + (p_b * model.k_d) / max(theta_layer, scalar(1e-03))
    else:
        print("No sorption assumed, Ret. factor = 1")
        retard_layer = 1  # No retardation.
        # For Freundlich parameters:
        # https://sitem.herts.ac.uk/aeru/ppdb/en/Reports/1027.htm
        # Retardation factor (eq 6): https://www.engr.mun.ca/~ccoles/Publications/ICWEM-023.pdf

    if gas:  # Leistra et al., 2001
        theta_gas = max(model.theta_sat[layer] - theta_layer, scalar(0))

        conc_aq = max(scalar(0), (mass / (cellarea() * depth)) /  # m2 * mm = L
                      (theta_gas / model.k_h + max(theta_layer, scalar(1e-03)) * retard_layer))  # mass/L cell volume
        # conc_aq = max(scalar(0), (mass / (cellarea() * depth)) /  # m2 * mm = L
        #               (max(theta_layer, scalar(1e-03)) * retard_layer))  # mass/L cell volume

        # conc_ads = model.k_d * conc_aq
        # mass_aq = conc_aq * (theta_layer * depth * cellarea())
        # if layer == 2:
        #     model.report(conc_aq, "Caq2")
        #     model.report(mass_aq, "Maq2")
        #     model.report(mass, "Mtot2")
        #     model.report(cellarea(), "cAr2")
        #     model.report(depth, "depth2")

    else:  # No gas phase
        # Whelan, 1987
        conc_aq = max(scalar(0), (mass / (cellarea() * depth)) / (max(theta_layer, scalar(1e-03)) * retard_layer))
        # conc_ads = model.k_d * conc_aq

    return conc_aq


def getLightMass(model, mass, app_indx):
    delta = mass / (1 + model.r_standard * (model.appDelta[app_indx] / 1000 + 1))
    return delta


def getVolatileMass(model, temp_air, mass,  # frac,
                    rel_diff_model="option-2", sorption_model="linear",
                    gas=True, run=True):
    if not run:
        volat_flux = model.zero_map
    else:  # Volatilize only during peak volatilization time i.e., first 24 hrs, @Prueger2005.
        layer = 0
        theta_gas = max(model.theta_sat[layer] - model.theta[layer], scalar(0))
        # Convert to m (needed for final mass computation on cell basis)
        depth_m = model.layer_depth[0] * 1 / 10 ** 3
        # Air boundary layer, assumed as 2m high
        thickness_a = scalar(1.0)  # m
        # Diffusion coefficient in air (cm^1/s); https://www.gsi-net.com
        #  D_ar (metolachlor) = 0.03609052694,  at reference Temp., in Kelvin, D_a,r)
        diff_ar = 0.03609052694 * 86400.0 * 1.0 / 10 ** 4  # m2/d
        # Diffusion coefficient adjusted to air Temp. in Kelvin, D_a
        diff_a = ((temp_air + 273.15) / 293.15) ** 1.75 * diff_ar  # m2/d

        if rel_diff_model == "option-2":
            # Millington and Quirk, 1960 (in Leistra, 2001, p.48)
            # a,b parameters: Jin and Jury, 1996 (in Leistra, 2001)
            diff_relative_gas = max((diff_a * theta_gas ** 2 /
                                     model.theta_sat[layer] ** (2 / 3)), scalar(1e10 - 6))  # m2/d
        elif rel_diff_model == "option-1":
            # Currie 1960 (in Leistra, 2001)
            # a,b parameters: Baker, 1987 (in Leistra, 2001)
            diff_relative_gas = max((diff_a * 2.5 * theta_gas ** 3), scalar(1e10 - 6))  # m2/d
        else:
            print("No appropriate relative diffusion parameter chosen")
            diff_relative_gas = diff_a  # m2/d
        # Transport resistance through air (r_a) and soil (r_s) layer
        r_a = thickness_a / diff_a  # d/m
        r_s = max(scalar(0), (0.5 * depth_m) / diff_relative_gas)  # d/m

        conc_aq = getConcAq(model, layer, mass, sorption_model=sorption_model, gas=gas)
        # Convert ug/L to ug/m3, as will be multiplying by cell's area in m2
        conc_aq *= 10 ** 3  # ug/L * 10^3 L/m3
        conc_gas = conc_aq / model.k_h  # ug/L air
        volat_flux = (conc_gas / (r_a + r_s)) * cellarea()  # ug/day
    return volat_flux


# Runoff
def getKfilm(model, runoffvelocity):
    """
    Note: Model uses run-off (mm) per day (i.e. timestep) as runoff velocity.
    Chemical parameter source:
    http://www.gsi-net.com/en/publications/gsi-chemical-database/single/377.html
    """
    # Dynamic viscosity of water (\mu) @25 Celsius = 8.9e-04 [Pa s]
    #   2 Pa = 2 N/(m s^1) = 2 Kg/(m s^1)
    #   Convert to g/(cm s): dyn_visc = 8.9e-03 [g/cm s]
    dyn_visc = 8.9e-03  # [g/cm s] @25 degrees, [@Shi2011]:\mu

    # Solute diffusivity in water (D_w)
    # Metolachlor = 5.0967719112e-006 (cm2 / s)
    diff_solute = 5.0967719112e-006  # [cm2 / s], [@Shi2011]:D_w
    Sc = dyn_visc / (model.p_bAgr * diff_solute)  # (-) Schmidt number, [@Shi2011]:S_c

    # Reynolds number (dimensionless), 86400s = 2 day
    cell_length = 2 * 10 ** 3  # mm
    # Reynolds (Re), [-] (Shi et al., 2011)
    re = (model.p_bAgr * 1 / 10 ** 2 * runoffvelocity * cell_length) / (dyn_visc * 86400)
    kl = (0.664 * ((diff_solute * 86400 * 10 ** 2) / cell_length) *
          re ** (float(1) / 2) * Sc ** (float(1) / 3))
    return kl  # mm/day


def getRunOffMass(model, precip, runoff_mm, mass,
                  transfer_model="simple-mt", sorption_model="linear",
                  gas=True, debug=False, run=True):
    if not run:
        mass_ro = deepcopy(model.zero_map)
    elif debug:
        mass_ro = deepcopy(model.zero_map)
    else:
        # Aqueous concentration
        layer = 0
        conc_aq = getConcAq(model, layer, mass, sorption_model=sorption_model, gas=gas)

        if transfer_model == "simple-mt":
            mass_ro = conc_aq * runoff_mm * cellarea()
        elif transfer_model == "nu-mlm-ro":
            # non-uniform-mixing-layer-model-runoff (nu-mlm-ro)
            # Considers a decrease in effective transfer as mixing layer depth increases
            # Adapted from Ahuja and Lehman, 1983 in @Shi2011,
            # Adaptation replaces Precip by Runoff amount.
            # beta_runoff = 1  # [mm] Calibration constant, 2 >= b > 0 (b-ranges appear reasonable).
            # As b decreases, mass transfer increases, model.z0 in mm
            mass_ro = conc_aq * (runoff_mm * cellarea()) * exp(-model.beta_runoff * model.layer_depth[layer])
        elif transfer_model == "nu-mlm":
            # non-uniform-mixing-layer-model (nu-mlm)
            # Original from Ahuja and Lehman, 1983 in @Shi2011
            # beta_runoff = 1 # [mm] Calibration constant, 2 >= b > 0 (b-ranges appear reasonable).
            # As b decreases, mass transfer increases, model.z0 in mm
            mass_ro = conc_aq * (precip * cellarea()) * exp(-model.beta_runoff * model.layer_depth[layer])
            mass_ro = ifthenelse(runoff_mm > scalar(0), mass_ro, scalar(0))
        elif transfer_model == "d-mlm":
            # distributed mixing-layer-model (d-mlm)
            # Adapted from Havis et al., 1992, and
            # taking the K_L definition for laminar flow from Bennett and Myers, 1982.
            mass_ro = getKfilm(model, runoff_mm) * cellarea() * conc_aq
        else:
            print("Run-off transfer model not stated")
            return None

        mReport = False
        if mReport:
            pass
            # model.report(conc_aq, 'aCo')
            # model.report(mass_ro, 'aMROa')
            # model.report(runoff_mm, 'aRO')

    return mass_ro


def getLeachedMass(model, layer, water_flux,
                   mass,
                   sorption_model=None,
                   leach_model=None, gas=True, debug=False, run=True):
    if not run:
        mass_leached = deepcopy(model.zero_map)
    elif debug:
        mass_leached = deepcopy(model.zero_map)
    else:
        if mapminimum(model.theta[layer]) < scalar(1e-06):
            mass_leached = deepcopy(model.zero_map)
        else:
            theta_layer = model.theta[layer]
            depth = model.layer_depth[layer]
            if layer < 2:
                p_b = model.p_bAgr
            else:
                p_b = model.p_bZ

            # Aqueous concentration
            conc_aq = getConcAq(model, layer, mass,
                                sorption_model=sorption_model, gas=gas)

            # Mass available for transport
            mass_aq = conc_aq * (theta_layer * depth * cellarea())

            # if layer == 2:
            #     model.report(conc_aq, "Caq")
            #     model.report(mass_aq, "Maq")
            #     model.report(mass, "Mtot")
            #     model.report(cellarea(), "cAr")
            #     model.report(depth, "depth1")

            test = mass - mass_aq
            if mapminimum(test) < 0:
                print("Error, mass < mass_aq, on layer: ", str(layer))
                model.report(test, 'aMzErr' + str(layer))

            if mapminimum(mass_aq) < 0:
                print("Corrected error caught in getLeachedMass(), mass_aq < 0")
                mass_aq = max(mass_aq, scalar(0))

            if sorption_model == "linear":
                # Retardation factor
                retard_layer = scalar(1) + (p_b * model.k_d) / theta_layer
            else:
                print("No sorption assumed, Ret. factor = 2")
                retard_layer = scalar(1)  # No retardation.

            if leach_model == "mcgrath":
                if layer == 0:
                    mass_aq_new = mass_aq * exp(-water_flux / (theta_layer * retard_layer * depth))
                    mass_leached = mass_aq - mass_aq_new
                    if mapminimum(mass_leached) < -1e-06:
                        print("Error in Leached Model, layer: ", str(layer))
                        model.report(mass_leached, 'aZ' + str(layer) + 'LCH')
                else:
                    # mass_aq_new = mass_aq * exp(-water_flux / (theta_layer * retard_layer * depth))
                    # mass_leached = mass_aq - mass_aq_new

                    # McGrath not used in lower layers,
                    # as formulation accounts for rainfall impact
                    max_flux = max(min(water_flux, (theta_layer - model.theta_fc[layer]) * depth), scalar(0))
                    mass_leached = conc_aq * max_flux * cellarea()

                    mass_aq = conc_aq * (theta_layer * depth * cellarea())
                    mass_aq_new = mass_aq - mass_leached
                    if mapminimum(mass_aq_new) < -1e-06:
                        print("Error in Leached Model, layer: ", str(layer))
                        model.report(mass_leached, 'aZ' + str(layer) + 'LCH')
                    if mapminimum(mass_aq_new) < 0:
                        print("Err mass_aq_new")
                        mass_leached = max(mass_leached, scalar(0))
                        # mass_aq_new = mass_aq - mass_leached
            else:
                mass_leached = conc_aq * water_flux * cellarea()
                mass_aq = conc_aq * (theta_layer * depth) * cellarea()
                mass_aq_new = mass_aq - mass_leached
                if mapminimum(mass_aq_new) < 0:
                    print("Error in Leached Model")

    return mass_leached


def getLatMassFlux(model, layer, mass, flux_map_mm,
                   sorption_model='linear', gas=True,
                   debug=False, run=True):
    """
    :param model:
    :param layer:
    :param mass:
    :param sorption_model:
    :param gas:
    :param debug:
    :param run:
    :return:
    """
    if not run:
        latflux_dict = {
            'mass_loss': deepcopy(model.zero_map),
            'mass_gain': deepcopy(model.zero_map),
            'new_mass': deepcopy(mass)
        }
    else:
        if mapminimum(model.theta[layer]) < scalar(1e-06) or layer == (model.num_layers - 1):
            latflux_dict = {
                'mass_loss': deepcopy(model.zero_map),
                'mass_gain': deepcopy(model.zero_map),
                'new_mass': deepcopy(mass)
            }
        else:

            # Aqueous concentration  mass/L
            conc_aq = getConcAq(model, layer, mass, sorption_model=sorption_model, gas=gas)

            mass_loss = conc_aq * flux_map_mm * cellarea()  # mm * m2 = L
            mass_gain = upstream(model.ldd_subs, mass_loss)
            new_mass = mass - mass_loss + mass_gain
            # http://pcraster.geo.uu.nl/pcraster/4.1.0/doc/manual/op_upstream.html

            if mapminimum(new_mass) < 0:
                print("Corrected error caught in getLatMassFlux(), new_mass < 0")
                new_mass = max(new_mass, scalar(0))

            if debug:
                model.report(mass, 'aMi' + str(layer))
                model.report(mass_loss, 'aMloss' + str(layer))
                model.report(mass_gain, 'aMgain' + str(layer))
                # aguila --scenarios='{2}' --timesteps=[2,300,2] aMi0 aMloss0 aMgain0

            latflux_dict = {
                'mass_loss': mass_loss,
                'mass_gain': mass_gain,
                'new_mass': new_mass
            }
    return latflux_dict


def getLatMassFluxManfreda(model, layer, mass, cell_moisture_outflow, upstream_cell_inflow,
                           sorption_model='linear', gas=True,
                           debug=False, run=True):
    """
    :param model:
    :param layer:
    :param mass:
    :param cell_moisture_outflow: mm
    :param upstream_cell_inflow: mm
    :param sorption_model:
    :param gas:
    :param debug:
    :param run:
    :return:
    """
    if not run:
        latflux_dict = {
            'mass_loss': deepcopy(model.zero_map),
            'mass_gain': deepcopy(model.zero_map),
            'net_mass_latflux': deepcopy(model.zero_map)
        }
    else:
        theta_layer = model.theta[layer]
        if mapminimum(model.theta[layer]) < scalar(1e-06) or layer == (model.num_layers - 1):
            latflux_dict = {
                'mass_loss': deepcopy(model.zero_map),
                'mass_gain': deepcopy(model.zero_map),
                'net_mass_latflux': deepcopy(model.zero_map)
            }
        else:
            depth = model.layer_depth[layer]
            c = model.c_lf[layer]

            # Aqueous concentration  ug/L
            conc_aq = getConcAq(model, layer, mass, sorption_model=sorption_model, gas=gas)

            # W(j/i)
            rel_wetness = model.wetness / accuflux(model.ldd_subs, model.wetness)
            # # Cell mass loss/gain (to update only mass)
            mass_loss = max(conc_aq * (c * (depth * theta_layer - depth * model.theta_fc[layer])), scalar(0))
            mass_gain = rel_wetness * accuflux(model.ldd_subs, mass_loss)

            # mass_gain = max(upstream(model.ldd_subs, conc_aq * upstream_cell_inflow * cellarea()), scalar(0))
            # mass_loss = max(downstream(model.ldd_subs, mass_gain), scalar(0))
            # http://pcraster.geo.uu.nl/pcraster/4.1.0/doc/manual/op_upstream.html
            net_mass_latflux = mass_gain - mass_loss

            if debug:
                model.report(mass, 'aMi' + str(layer))
                model.report(mass_loss, 'aMloss' + str(layer))
                model.report(mass_gain, 'aMgain' + str(layer))
                # aguila --scenarios='{2}' --timesteps=[2,300,2] aMi0 aMloss0 aMgain0

            latflux_dict = {
                'mass_loss': mass_loss,
                'mass_gain': mass_gain,
                'net_mass_latflux': net_mass_latflux
            }
    return latflux_dict


def getDrainMassFlux(model, layer, mass,
                     sorption_model='linear', gas=True,
                     debug=False, run=True):
    if run:
        # Aqueous concentration
        conc_aq = getConcAq(model, layer, mass, sorption_model=sorption_model, gas=gas)
        cell_mm = max( model.c_adr * (model.layer_depth[layer] * model.theta[layer] -
                                                               model.layer_depth[layer] * model.theta_fc[layer]),  scalar(0))
        mass_loss = conc_aq * cellarea() * cell_mm
    else:
        mass_loss = deepcopy(model.zero_map)

    return mass_loss


def getMassDegradation(model, layer, mass, old_aged_mass,
                       frac="L",
                       sor_deg_factor=1,
                       sorption_model="linear", fixed_dt50=True, deg_method=None,
                       bioavail=True, gas=True,
                       debug=False, run=True):
    if not run:
        return {"mass_tot_new": deepcopy(mass),
                "mass_deg_aq": deepcopy(model.zero_map),
                "mass_deg_ads": deepcopy(model.zero_map)}
    else:
        theta_wp = model.theta_wp[layer]
        theta_layer = model.theta[layer]
        theta_gas = max(model.theta_sat[layer] - theta_layer, scalar(0))
        depth = model.layer_depth[layer]
        if layer < 2:
            p_b = model.p_bAgr
        else:
            p_b = model.p_bZ

        # Step 0 - Obtain species concentration (all phases)
        conc_aq = getConcAq(model, layer, mass,
                            sorption_model=sorption_model, gas=gas)  # mass/L
        conc_ads = model.k_d * conc_aq
        # conc_ads = getConcAds(model, layer, bioa_mass, gas=gas)  # mass/g soil

        mass_aq = conc_aq * (theta_layer * depth * cellarea())
        mass_ads = conc_ads * (p_b * depth * cellarea())  # pb = g/cm3
        # Check gas
        mass_gas = max(mass - mass_aq - mass_ads, scalar(0))

        if bioavail:
            # Mass compartment (bio-available fraction)
            k_aged = ln(2) / model.dt_50_aged
            bioa_mass = mass_ads * exp(-k_aged * scalar(model.jd_dt))
            aged_mass = mass_ads - bioa_mass
        else:
            bioa_mass = deepcopy(mass_ads)
            aged_mass = deepcopy(model.zero_map)
        #
        k_ab = max(ln(2) / model.dt_50_ab, scalar(0))
        total_aged = old_aged_mass + aged_mass
        mass_aged_new = total_aged * exp(-k_ab * scalar(model.jd_dt))
        mass_deg_aged = total_aged - mass_aged_new

        # tot_ba_mass = mass_aq + mass_ads + mass_gas
        # error = tot_ba_mass - mass
        # model.report(error, frac + 'errZ' + str(layer))

        if not fixed_dt50:
            # F_Theta_1
            if deg_method == 'schroll':  # Schroll et al., 2006
                theta_factor = ifthenelse(theta_layer <= 0.5 * theta_wp, scalar(0),
                                          ifthenelse(theta_layer <= model.theta_100[layer],
                                                     (((theta_layer - 0.5 * theta_wp) / (
                                                         model.theta_100[layer] - theta_wp)) ** scalar(
                                                         model.beta_moisture)),
                                                     scalar(1)))
            else:  # Walker, 1973, Macro
                assert float(model.theta_ref) > 0

                theta_factor = (theta_layer / model.theta_ref) ** scalar(model.beta_moisture)

            temp_factor = exp(
                (model.act_e / model.r_gas) * (1 / (model.temp_ref + 273.15) - 1 / (model.temp_air + 273.15)))

            # Half-life as a function of temperature and moisture
            dt_50 = max(model.dt_50_ref * theta_factor * temp_factor, scalar(0))
        else:
            dt_50 = model.dt_50_ref

        # Convert to degradation constant
        # Deg in dissolved phase
        k_b = ifthenelse(dt_50 == scalar(0), scalar(0),
                         ln(2) / dt_50)  # Constant of degradation (-) is dynamic, based on Theta and Temp.

        # Deg in sorbed phase (now assumed equal)
        k_bs = k_b * sor_deg_factor

        # Step 2 - Degrade phase fractions
        # First order degradation kinetics
        if frac == "H":
            mass_aq_new = mass_aq * exp(-1 * model.alpha_iso * k_b * scalar(model.jd_dt))
            mass_ads_new = bioa_mass * exp(-1 * model.alpha_iso * k_bs * scalar(model.jd_dt))
        else:  # Same for total conc as for "L", but without alpha
            mass_aq_new = mass_aq * exp(-1 * k_b * scalar(model.jd_dt))
            mass_ads_new = bioa_mass * exp(-1 * k_bs * scalar(model.jd_dt))

        # Step 1 - Convert back to mass (i.e., after degradation in each phase)
        # mass_aq_new = conc_aq_new * (theta_layer * depth * cellarea())
        # mass_ads_new = conc_ads_new * (model.p_b * depth * cellarea())  # pb = g/cm3
        # mass_gas = conc_gas * (theta_gas * depth * cellarea())
        mass_tot_new = mass_aq_new + mass_ads_new + mass_gas  # + aged_mass
        mass_deg_aq = mass_aq - mass_aq_new
        mass_deg_ads = bioa_mass - mass_ads_new
        # if frac == "L" and layer == 0:
        #     model.report(mass, 'adt0M')
        #     model.report(mass_tot_new, 'adt1M')
        #   # model.report(mass_aq_new, 'adt1M')
        if debug:
            model.report(bioa_mass, 'aBIOz' + str(layer))
            model.report(theta_factor, 'aFz' + str(layer))
            model.report(temp_factor, 'aTz' + str(layer))
            model.report(dt_50, 'aDT50z' + str(layer))
            model.report(k_b, 'k_b' + str(layer))
            model.report(k_ab, 'k_ab' + str(layer))
            model.report(k_aged, 'k_age' + str(layer))
            model.report(theta_layer, 'thd' + str(layer))
            model.report(model.theta_ref, 'thref' + str(layer))
            model.report(model.beta_moisture, 'BetaZ' + str(layer))
            # aguila --scenarios='{1}' --timesteps=[1,300,1] aFz aTz aDT50z k_b thd0 thref BetaZ
            # aguila --scenarios='{1}' --timesteps=[1,300,1]  k_b k_ab thd0 thref BetaZ aFz
            # aguila --scenarios='{1}' --timesteps=[1,300,1]  k_b k_ab k_age aBIOz0

    # if frac == "H":
    #     model.report(mass_tot_new, 'HinZ' + str(layer))
    # else:
    #     model.report(mass_tot_new, 'LinZ' + str(layer))

    return {"mass_tot_new": mass_tot_new,
            "mass_deg_aq": mass_deg_aq,
            "mass_deg_ads": mass_deg_ads,
            # "aged_mass": aged_mass,
            "mass_aged_new": mass_aged_new,
            "mass_deg_aged": mass_deg_aged}
