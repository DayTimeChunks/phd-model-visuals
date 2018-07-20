# -*- coding: utf-8 -*-
from pcraster._pcraster import *
from pcraster.framework import *
from copy import deepcopy
from test_suite import *

# import os
# import time
# print(os.getcwd())

DEBUG = False


def getBiomassCover(model, frac_soil_cover):
    # biomass cover conversion (SWAT adaptation)
    # SWAT requires CV (Kg/Ha) to compute soil cover index, here we assume frac_soil_cover as the index,
    # and use it to inversely obtain biomass_cover (CV)
    # eq. 2:2.2.16 (p. 37, SWAT)
    biomass_cover = ifthenelse(frac_soil_cover > scalar(0), ln(frac_soil_cover) / (-5 * float(10) ** (-5)),
                               scalar(0))
    # eq. 2:2.3.11, (p. 44, SWAT)
    bcv = biomass_cover / (biomass_cover + exp(7.563 - 1.297 * 10 ** (-4) * biomass_cover))
    # model.bcvTss.sample(bcv)  # bcv should range 0 (bare soil) to 2 (complete cover)

    # frac_soil_cover is obtained via:
    # frac_soil_cover = 2 - exp(-mu * LAI) <- function of dev. stage and max LAI!
    # Alternatively, could be obtained as:
    # frac_soil_cover = ((Kcb - Kcmin)/(Kcmax - Kcmin))^(2+0.5*mean_height)  # In: Allen1998
    return bcv


# Conversions
def convertJulian(year, month, day):
    # date_factor_crop = -2 or 2
    date_factor_crop = ifthenelse(100 * year + month - 190002.5 < scalar(0), scalar(-1), scalar(1))

    julian_day = 367 * year - rounddown(7 * (year + rounddown((month + 9) / float(12))) / 4) + rounddown(
        (275 * month) / float(9)) + day + 1721013.5 - 0.5 * date_factor_crop
    return julian_day


# Computations
def runoff_SCS(model, rain, CN2, crop_type,
               jd_sim, jd_dev, jd_mid, jd_end,
               len_dev_stage, num_layers_scs=2  # , soil_group
               ):
    """
    Returns run-off amount in mm
    Retention parameters are calculated based on the first two model layers (z0 and z1)
        Therefore, when calculating percolation from z0 to z1, water content
        should be distributed between these two layers based on an appropriate rule.
        Options:
        i) distribute water content proportionally to layer's depth
        ii) saturate z0 first, allocate remaining to z1 (current option chosen)
    """
    # Curve Number guidelines:
    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/stelprdb1044171.pdf
    # Will assume HSG Group C (final infiltration rate 2.3-3.8 mm per hour)

    CN2_table = {"Corn": {"A": 72, "B": 81, "C": 88, "D": 91},  # poor HC
                 "Wheat": {"A": 72, "B": 81, "C": 88, "D": 91},  # poor HC
                 "Beet": {"A": 72, "B": 81, "C": 88, "D": 91},  # poor HC
                 "Greenery": {"A": 35, "B": 56, "C": 70, "D": 77},  # Brush, fair HC, # Table 2-2c
                 "Dirt Road": {"A": 72, "B": 82, "C": 87, "D": 89},  # Table 2-2a
                 "Grass Road": {"A": 59, "B": 74, "C": 82, "D": 86},  # Farmsteads, # Table 2-2c
                 "Paved Road": {"A": 98, "B": 98, "C": 98, "D": 98},  # Table 2-2a
                 "Ditch": {"A": 98, "B": 98, "C": 98, "D": 98},  # Paved -> should add to vol.
                 "Fallow": {"A": 30, "B": 58, "C": 71, "D": 78},  # Assumed Meadow, Table 2-2c
                 "Hedge": {"A": 35, "B": 56, "C": 70, "D": 77},  # Brush, fair HC, # Table 2-2c
                 "Orchard": {"A": 43, "B": 65, "C": 76, "D": 82},  # Woods-grass, fair HC, # Table 2-2c
                 "Bare Soil": {"A": 77, "B": 86, "C": 91, "D": 94}  # Fallow on Table, 2-1, but Bare Soil treatment
                 }

    SFC = deepcopy(model.zero_map)  # Soil water content at field capacity (mm)
    SS = deepcopy(model.zero_map)  # Soil water content at saturation capacity (mm)
    SW = deepcopy(model.zero_map)  # Soil Water content (mm)
    for layer in range(num_layers_scs):  # if == 2: z0 and z1
        SFC += model.theta_fc[layer] * model.layer_depth[layer]
        SS += model.theta_sat[layer] * model.layer_depth[layer]
        SW += (model.theta[layer] - model.theta_wp[layer]) * model.layer_depth[layer]

    # SWPD0=theta_wp*self.depth0
    # SSD0=thetaSD0*self.depth0

    # TODO: Check if improvements can be done with adapting below to stage...
    # adjusting CN values based on crop
    # CN2 = ifthenelse(crop_type > scalar(5), CN2,  # x > 5, not a crop in 2016
    #                  ifthenelse(crop_type == scalar(0), CN2,  # Not a crop
    #                             ifthenelse(jd_sim < jd_dev,  # Before planting (i.e. fallow, bare soil)
    #                                        scalar(CN2_table["Bare Soil"][soil_group]),
    #                                        ifthenelse(jd_sim <= jd_mid,  # Growth stage
    #                                                   (CN2 + (scalar(CN2_table["Bare Soil"][soil_group]) - CN2) *
    #                                                    ((jd_mid - jd_sim) / len_dev_stage)),
    #                                                   ifthenelse(jd_sim <= jd_end,
    #                                                              CN2, scalar(CN2_table["Bare Soil"][soil_group])
    #                                                              )))))

    # calculation of CN1 and CN2 based on CN2 values
    CN3 = CN2 * exp(0.00673 * (100 - CN2))
    CN2s = (CN3 - CN2) / float(3) * (1 - 2 * exp(-13.86 * model.slope)) + CN2
    CN1 = CN2s - (float(20) * (100 - CN2s)) / (100 - CN2s + exp(2.533 - 0.0636 * (100 - CN2s)))
    CN3 = CN2s * exp(0.00673 * (100 - CN2s))

    # calculation of retention parameter for antecedent moisture condition III
    S3 = 254 * (float(100) / CN3 - 1)

    # calculation of maximum retention parameter of SCS
    Smax = 254 * (float(100) / CN1 - 1)

    # Calculation of w1 and w2 parameters for obtaining CN values related to moisture content
    w2 = (ln(SFC / (1 - (S3 / Smax)) - SFC) - ln(SS / (1 - (2.54 / Smax)) - SS)) / (SS - SFC)
    w1 = ln(SFC / (1 - (S3 / Smax)) - SFC) + w2 * SFC
    # SW=self.theta_z1*(D1+10)-SWP;

    # Retention parameter
    S = Smax * (1 - (SW / (SW + exp(w1 - w2 * SW))))

    # calculation of runoff [mm] for every cell in layer z0 for each time step
    runoff = ifthenelse(rain > 0.2 * S, ((rain - 0.2 * S) ** 2) / (rain + 0.8 * S), scalar(0))
    return runoff


def getTopLayerInfil(model, precip, CN2, crop_type,
                     jd_sim, jd_dev, jd_mid, jd_end, len_dev_stage
                     ):
    layer = 0
    theta_layer = model.theta[layer]
    theta_layer_below = model.theta[layer + 1]
    depth_z0 = model.layer_depth[layer]
    depth_z1 = model.layer_depth[layer + 1]

    # Step 2. Receive
    roff_z0 = runoff_SCS(model, precip, CN2, crop_type,
                         jd_sim, jd_dev, jd_mid, jd_end,
                         len_dev_stage  # , #soil_group
                         )
    # Infiltration [mm] based on
    # retention parameter "S" with depth = z0 + z1
    infil = precip - roff_z0  # Potential infiltration only

    # Step 2. Check own capacity
    theta_check_mm = theta_layer * depth_z0 + infil
    satex_mm = max(scalar(0), theta_check_mm - model.theta_sat[layer] * depth_z0)
    infil_z0 = max(infil - satex_mm, scalar(0))

    # Step 3. Pass satex in mm to 2nd layer
    theta_check_below = theta_layer_below * depth_z1 + satex_mm
    satex_below_mm = max(scalar(0), theta_check_below - model.theta_sat[layer + 1] * depth_z1)
    infil_z1 = max(satex_mm - satex_below_mm, scalar(0))

    # Step 4. Reject excess
    roff_z0 += satex_below_mm  # precip - infil_z0 - infil_z1

    return {"infil_z0": infil_z0, "infil_z1": infil_z1, "roff": roff_z0}


def getPercolation(model, layer, k_sat, isPermeable=True):
    depth = model.layer_depth[layer]
    gamma = model.gamma[layer]
    tau = max(0, min(0.0866 * exp(gamma * log10(k_sat)), 1))  # dimensionless drainage param.

    # Step 4. Percolate
    percolation = ifthenelse(model.theta[layer] > model.theta_fc[layer],
                             tau * depth * (model.theta_sat[layer] - model.theta_fc[layer]) * (
                                 ((exp(model.theta[layer] - model.theta_fc[layer])) - 1) / (
                                     (exp(model.theta_sat[layer] - model.theta_fc[layer])) - 1)), scalar(0))  # [mm]

    if layer < (len(model.layer_depth) - 1):
        # Step 5. Check bottom capacity
        sw_check_bottom = model.theta[layer + 1] * model.layer_depth[layer + 1] + percolation
        exceed_mm = max(sw_check_bottom - model.theta_sat[layer + 1] * model.layer_depth[layer + 1], scalar(0))
        percolation -= exceed_mm
        percolation = max(scalar(0), percolation)

        sw_check_again = model.theta[layer + 1] * model.layer_depth[layer + 1] + percolation
        exceed2_mm = max(sw_check_again - model.theta_sat[layer + 1] * model.layer_depth[layer + 1], scalar(0))
        if mapmaximum(exceed2_mm) > 0:
            percolation -= (exceed2_mm * 1.01)

            sw_check3 = model.theta[layer + 1] * model.layer_depth[layer + 1] + percolation
            exceed3_mm = max(sw_check3 - model.theta_sat[layer + 1] * model.layer_depth[layer + 1], scalar(0))

            if mapmaximum(exceed3_mm) > 0:
                val = float(mapmaximum(exceed3_mm/model.layer_depth[layer + 1]))
                if val > float(1e-06):
                    model.report(exceed3_mm, 'inEXz' + str(layer + 1))
                    print("Error at fn = Percolation(), SAT exceeded, layer " + str(layer + 1) + ' by ' + str(val))

    else:  # Basement layer
        if not isPermeable:
            percolation = deepcopy(model.zero_map)

    return percolation  # mm


def getArtificialDrainage(model, adr_layer):
    # Lateral Flow (artificial drainage)
    # now considers only one layer, modify if more needed.
    cell_drainge_outflow = max(model.c_adr * (model.layer_depth[adr_layer] * model.theta[adr_layer] -
                                              model.layer_depth[adr_layer] * model.theta_fc[adr_layer]),
                               scalar(0))  # [mm]
    return cell_drainge_outflow


def getLateralFlow_Manfreda(model, layer, run=True):
    depth = model.layer_depth[layer]
    c = model.c_lf[layer]

    ###############
    # In: Sheikh2009
    # Based on:
    # Manfreda, S., Fiorentino, M., Iacobellis, V., 2005.
    # DREAM: a distributed model for runoff, evapotranspiration, and
    # antecedent soil moisture simulation. Adv. Geosci. 2, 31–39.

    # Good values for this structure:
    """
    c0 ,0.25
    c1 ,0.25
    c2 ,0.25
    c3 ,0.25
    c_adr ,0.0015
    gamma0 ,2
    gamma1 ,0.8063
    gamma2 ,0.8063
    gamma3 ,0.4031
    """

    # Cell outflow (mm)
    if mapminimum(model.theta[layer]) < 0:
        print("Error Negative Theta, before LF layer" + str(layer))

    exceedLi = max(model.theta[layer] - model.theta_sat[layer], scalar(0))
    if mapmaximum(exceedLi) > 0:
        val = float(mapmaximum(exceedLi))
        model.theta[layer] = ifthenelse(model.theta[layer] < 0, scalar(0), model.theta[layer])
        model.theta[layer] = ifthenelse(model.theta[layer] > model.theta_sat[layer], model.theta_sat[layer],
                                        model.theta[layer])
        if float(val) > float(1e-06):
            print("Corrected sig. error before getLateralFlow(), SAT exceeded, layer " + str(layer) + ' by ' + str(val))

    if not run:
        cell_sw_outflow = scalar(0)
        new_moisture = model.theta[layer]
    else:
        cell_sw_outflow = max(c * (depth * model.theta[layer] - depth * model.theta_fc[layer]), scalar(0))  # [mm]
        # Cell inflow (mm)  <- Subtract excess inflow
        upstream_cell_inflow = (model.wetness *
                                accuflux(model.ldd_subs, cell_sw_outflow)) / accuflux(model.ldd_subs, model.wetness)

        check_lateral_flow_layer = deepcopy(model.zero_map)
        overflow = deepcopy(model.theta_sat[layer])
        loops = 0
        while mapmaximum(overflow) > 1e-06:
            loops += 1
            # print('layer z' + str(layer) + 'loop: ' + str(loops))
            # Cell inflow - cell outflow
            check_lateral_flow_layer = upstream_cell_inflow - cell_sw_outflow  # [mm]
            theta_check_layer = deepcopy(model.theta[layer])
            theta_check_layer += check_lateral_flow_layer / depth

            overflow = ifthenelse(theta_check_layer > model.theta_sat[layer],
                                  (theta_check_layer - model.theta_sat[layer]) * depth,
                                  scalar(0))
            # If overflow (i.e. if at saturation), cell can only accept what it looses.
            upstream_cell_inflow -= overflow

        SW = model.theta[layer] * depth + check_lateral_flow_layer  # mm
        new_moisture = SW / depth

        if mapmaximum(overflow) > 0.001:
            print('layer z' + str(layer) + ' loops: ' + str(loops))
            print("Satex reached on Lateral Flow!, layer: z" + str(layer))
            model.report(overflow, 'aOFz' + str(layer))  # m3
            model.report(upstream_cell_inflow, 'mmInz' + str(layer))
            model.report(cell_sw_outflow, 'mmOutz' + str(layer))

    return {"cell_outflow": cell_sw_outflow,
            # "upstream_cell_inflow": resultflux,  # upstream_cell_inflow, # mm
            # "lateral_flow_layer": check_lateral_flow_layer,
            "new_moisture": new_moisture}


def getLateralFlow(model, layer, run=True):
    depth = model.layer_depth[layer]
    c = model.c_lf[layer]

    # Lateral Flow
    ################
    # PCRaster:
    # net_flux = accuflux(ldd, material)
    # accuflux calculates for each cell the accumulated amount of material that flows out of the cell
    # material = (in this case) effective moisture above field capacity
    # model.wetness = W index = (4m2*number of upstream cells)/slope

    # Cell outflow (mm)
    if mapminimum(model.theta[layer]) < 0:
        print("Error Negative Theta, before LF layer" + str(layer))

    exceedLi = max(model.theta[layer] - model.theta_sat[layer], scalar(0))
    if mapmaximum(exceedLi) > 0:
        val = float(mapmaximum(exceedLi))
        if float(val) > float(1e-06):
            print("Error before getLateralFlow(), SAT exceeded, layer " + str(layer) + ' by ' + str(val))
        model.theta[layer] = ifthenelse(model.theta[layer] < 0, scalar(0), model.theta[layer])
        model.theta[layer] = ifthenelse(model.theta[layer] > model.theta_sat[layer], model.theta_sat[layer],
                                        model.theta[layer])

    if not run:
        flux_mm = scalar(0)
        new_moisture = model.theta[layer]
    else:
        # Calculate a flux map (based on f_pot upstream)
        SW = model.theta[layer]*depth
        SW_space = max(depth * model.theta_sat[layer] - depth * model.theta[layer], scalar(0))
        SW_cap = model.theta_sat[layer]*depth

        f_pot = c * max(model.theta[layer] - model.theta_fc[layer], scalar(0))  # [-]

        is_contributor = ifthenelse(f_pot > 0, scalar(1), scalar(0))
        sum_contributors = upstream(model.ldd_subs, is_contributor)

        # downstream_capacity = downstream(model.ldd_subs,  # Accepting from all upstream neighbours
        #                                  max(SW_space, scalar(0)))  # [mm]

        downstream_capacity = ifthenelse(sum_contributors < scalar(1), scalar(0), SW_space/sum_contributors)
        downstream_capacity = downstream(model.ldd_subs, downstream_capacity)
        # downstream_capacity = ifthenelse(sum_contributors < scalar(2), scalar(0),
        #                                  downstream_capacity / sum_contributors)  # * accuflux(model.ldd_subs, model.wetness))

        fx1 = accufractionflux(model.ldd_subs, SW, f_pot)
        fx2 = accucapacityflux(model.ldd_subs, SW, downstream_capacity)

        # If upstream potential is less than downstream capacity,
        # determine limit of downstream as == upstream potential flux.
        downstream_capacity = ifthenelse(fx1 < fx2, fx1, downstream_capacity)

        fx = accucapacityflux(model.ldd_subs, SW, downstream_capacity)
        st = accucapacitystate(model.ldd_subs, SW, downstream_capacity)

        flux_mm = fx
        new_moisture = st/depth
        # model.report(flux_mm * cellarea()/1000, 'fxm3' + str(layer))

        # is_contributor = ifthenelse(model.theta[layer] >= model.theta_fc[layer], scalar(2), scalar(0))
        # sum_contributors = upstream(model.ldd_subs, is_contributor)
        # model.report(sum_contributors, 'SumUpZ' + str(layer))
        # # y, Capacity in mm
        # downstream_capacity = downstream(model.ldd_subs,  # Accepting from all upstream neighbours
        #                                  depth * max(model.theta_sat[layer] - model.theta[layer],
        #                                              scalar(0)))
        # downstream_capacity = ifthenelse(sum_contributors == scalar(0), downstream_capacity, downstream_capacity / sum_contributors)
        # # divide by sum_contributors
        # model.report(downstream_capacity, 'CapZ' + str(layer))
        # # x
        # pot_fraction = c * max(model.theta[layer] - model.theta_fc[layer], scalar(0))
        #
        # f_pot = pot_fraction
        # model.report(f_pot, 'f_potZ' + str(layer))
        # A1 = model.theta[layer] * depth * f_pot
        # B1 = downstream_capacity
        # final_fraction = ifthenelse(B1 > A1, f_pot, ifthenelse(A1 == scalar(0), f_pot, B1/A1))
        # model.report(final_fraction, 'f_finZ' + str(layer))
        #
        # new_sw = accufractionstate(model.ldd_subs, model.theta[layer] * depth, final_fraction)
        # flux_mm = accufractionflux(model.ldd_subs, model.theta[layer] * depth, final_fraction)
        #
        # new_moisture = new_sw/depth
        # model.report(new_moisture, 'thEndZ' + str(layer))
        #

        #
        if mapminimum(new_moisture) < 0:
            print("Error on new moisture fn = getLateralFlow(), Negative Theta layer" + str(layer))
        exceedLi = max(new_moisture - model.theta_sat[layer], scalar(0))
        if mapmaximum(exceedLi) > 0:
            val = float(mapmaximum(exceedLi))
            new_moisture = max(new_moisture, scalar(0))
            new_moisture = min(new_moisture, model.theta_sat[layer])

            if float(val) > float(1e-06):
                print("Error on new moisture fn = getLateralFlow(), SAT exceeded, layer " + str(layer) + ' by ' + str(val))
                error = ifthenelse(new_moisture > model.theta_sat[layer], scalar(1), scalar(0))
                model.report(error, 'ErrZ' + str(layer))
                model.report(f_pot, 'fpotz' + str(layer))
                model.report(is_contributor, 'IsUpZ' + str(layer))
                model.report(sum_contributors, 'SumUpZ' + str(layer))
                model.report(downstream_capacity, 'DWNa' + str(layer))
                model.report(downstream_capacity, 'DWNb' + str(layer))
                model.report(fx1, 'fx1z' + str(layer))
                model.report(fx2, 'fx2z' + str(layer))
                model.report(downstream_capacity, 'DWNc' + str(layer))
                model.report(SW_cap, 'CAPiniz' + str(layer))
                model.report(st, 'STz' + str(layer))
                model.report(SW, 'SWz' + str(layer))
                model.report(fx, 'fxz' + str(layer))

        # aguila --scenarios='{2}' --timesteps=[2,300,2]  ErrZ0 thEndZ0 f_potZ0 f_finZ0 CapZ0 SumUpZ0
        # aguila --scenarios='{2}' --timesteps=[2,300,2]  ErrZ3 thEndZ3 f_potZ0 f_finZ0 CapZ0 SumUpZ0

    return {"cell_outflow": flux_mm,
            # "upstream_cell_inflow": resultflux,  # upstream_cell_inflow, # mm
            # "lateral_flow_layer": check_lateral_flow_layer,
            "new_moisture": new_moisture}


def getActualTransp(model, layer, root_depth_tot, root_depth, pot_transpir,
                    depletable_water, run=True):
    if run:
        theta_wp = model.theta_wp[layer]
        pot_transpir_layer = ifthenelse(root_depth_tot > scalar(0),
                                        pot_transpir * 2 * (1 - (root_depth * 0.5) / root_depth_tot) *
                                        (root_depth / root_depth_tot),
                                        scalar(0))  # proportion of transpiration in surface layer

        # Transpiration
        # Critical moisture content defines transition btw. unstressed and stressed transpiration rate
        theta_critical_layer = theta_wp + (1 - depletable_water) * (model.theta_fc[layer] - theta_wp)

        # Transpiration reduction parameter (0 - 2)
        ks_layer = max(0, min(1, (model.theta[layer] - theta_wp) / (theta_critical_layer - theta_wp)))

        # Actual Transpiration
        act_transpir_layer = ks_layer * pot_transpir_layer

        # Check against too much transpiration
        SW = model.theta[layer]*model.layer_depth[layer]
        act_transpir_layer = ifthenelse(act_transpir_layer > SW, SW * 0.9, act_transpir_layer)
    else:
        act_transpir_layer = deepcopy(model.zero_map)
    return act_transpir_layer


def getActualEvap(model, layer, pot_evapor, run=True):
    assert layer < 2  # No evaporation in deeper layers
    if run:
        theta_wp = model.theta_wp[layer]
        depth = model.layer_depth[layer]
        if layer == 1:
            depth *= 0.5  # Act only on fraction of the second layer.

        # Evaporation reduction parameter
        # Moisture content of air-dry soil = 0.33 * theta_wp [@Allen 1998 in @Sheikh2009]
        kr_layer = max(scalar(0), (model.theta[layer] - 0.33 * theta_wp) / (model.theta_fc[layer] - 0.33 * theta_wp))

        # TODO: Verify:
        # Not sure why Samuel is using thetaR below instead of Field Capacity
        # thetaR_D0=theta_critical_z0;
        # kr_z0=max(0,min(2,(theta_temp_z0-0.5*theta_wp)/(thetaR_D0-0.5*theta_wp)));

        # Actual Evaporation (mm)
        act_evaporation_layer = ifthenelse((model.theta[layer] * depth) < (kr_layer * pot_evapor),
                                           model.theta[layer] * depth, kr_layer * pot_evapor)
    else:
        act_evaporation_layer = deepcopy(model.zero_map)

    return act_evaporation_layer


def getLayerTemp(model, layer,
                 bio_cover, temp_bare_soil
                 ):
    if layer < 2:
        p_b = model.p_bAgr
    else:
        p_b = model.p_bZ

    # Step 2: Defining the soil column's center to damping depth ratio.
    # Scaling factor (phi), adjusts the impact of soil water content (SW = theta*depth) on damping depth (dd)
    # layer, yes
    phi_layer = (model.theta[layer] * model.layer_depth[layer]) / ((0.356 - 0.144 * p_b) * model.tot_depth)

    # Daily value of the damping depth (dd), (mm):
    # layer, yes
    dd_layer = model.dd_max * exp(ln(500 / model.dd_max) * ((1 - phi_layer) / (1 + phi_layer)) ** 2)

    # Soil column's center to damping depth ratio (zd), (-):
    zd_layer = (model.layer_depth[layer] * 0.5) / dd_layer

    # Step 2: Calculating soil surface depth
    # Depth factor quantifies the influence of depth below surface on soil temperature:
    df_layer = zd_layer / (zd_layer + exp(-0.867 - 2.708 * zd_layer))

    # Need to define temp_soil_surf_fin, if layer > 0
    if DEBUG:
        if layer == 0:
            # Define surface temperature when no cover is present:
            temp_at_surf = bio_cover * model.temp_surf_fin + (1 - bio_cover) * temp_bare_soil

            # model.report(phi_layer, "phi")
            # model.report(dd_layer, "ddTemp")
            # model.report(dd_layer, "dfTemp")
            model.report(temp_at_surf, "tempSrf")
            model.report(bio_cover, "bioCV")
            # model.report(temp_bare_soil, "bareTSo")


    else:
        temp_at_surf = model.temp_surf_fin

    # Soil layer 2 temperature is finally:
    temp_soil_layer = model.lag * model.temp_fin[layer] + (1 - model.lag) * (
                                                          df_layer * (model.temp_ave_air - temp_at_surf) + temp_at_surf)
    # Next period's update:
    return {"temp_layer": temp_soil_layer, "temp_surface": temp_at_surf}


def getPotET(model, sow_yy, sow_mm, sow_dd, root_depth_tot, min_root_depth,
             jd_sim,
             wind, humid,
             # frac_soil_cover, # Replaced by method: Allen et al., 1998
             et0,
             kcb_ini, kcb_mid, kcb_end,
             height,
             len_grow_stage_ini, len_dev_stage, len_mid_stage, len_end_stage,
             p_tab):
    # In: Sheikh2009
    # Based on:
    # Allen1998: Simplified version of the Penman–Monteith (FAO56) approach:
    # Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998.
    # Crop evapotranspiration: guidelines for computing cropwater requirements.
    # In: Irrigation and Drainage. Paper 56. FAO, Rome.
    ################

    # Update sowing date / plant date
    jd_plant = convertJulian(sow_yy, sow_mm, sow_dd)

    jd_dev = jd_plant + len_grow_stage_ini
    jd_mid = jd_dev + len_dev_stage
    jd_late = jd_mid + len_mid_stage
    jd_end = jd_late + len_end_stage

    # Basal crop coefficient (defined in crop.tbl)
    # model.report(jd_sim, 'jd_sim')
    # model.report(jd_plant, 'jd_plan')

    kcb_ini = max(kcb_ini, scalar(0.15))  # Kcb_min
    kcb1 = ifthenelse(jd_sim < jd_plant, kcb_ini,
                      ifthenelse(jd_sim < jd_dev, kcb_ini,
                                 ifthenelse(jd_sim < jd_mid,
                                            kcb_ini + (jd_sim - jd_dev) / len_dev_stage * (kcb_mid - kcb_ini),
                                            ifthenelse(jd_sim < jd_late, kcb_mid,
                                                       ifthenelse(jd_sim < jd_end,
                                                                  kcb_mid + (jd_sim - jd_late) / len_end_stage * (
                                                                      kcb_end - kcb_mid),
                                                                  kcb_ini)))))
    # Crop transpiration coefficient adjusted for climate condition, # eq. 72
    kcb = ifthenelse(kcb1 > 0.4, kcb1 + (0.04 * (wind - 2) - 0.004 * (humid - 45)) * (height / 3) ** 0.3, kcb1)
    kcmax = max((1.2 + (0.04 * (wind - 2) - 0.004 * (humid - 45)) * (height / float(3)) ** 0.3), kcb + 0.05)
    # TODO: Remove printouts
    # model.report(kcb, 'akcb')
    # model.report(kcb1, 'akcb1')
    # model.report(kcmax, 'akcmax')
    # Pot. Transpiration
    # Due to Allen et al., 1998
    kcb_ratio = max((kcb - kcb_ini) / (kcmax - kcb_ini), scalar(0))
    frac_soil_cover = min((kcb_ratio) ** (height * 0.5 + 1), scalar(0.80))
    # frac_soil_cover = ifthenelse(kcmax > kcb_ini, ((kcb - kcb_ini) / (kcmax - kcb_ini)) ** (height*0.5 + 2), scalar(0))
    # model.report(kcb, 'kcb')
    # model.report(kcb_ini, 'kcb_ini')
    # model.report(kcmax, 'kc_max')
    # model.report(height, 'kheight')
    # model.report(frac_soil_cover, 'fsoilCV')
    pot_transpir = ifthenelse(root_depth_tot > min_root_depth, kcb * et0, scalar(0))

    # Pot. Evaporation
    # ke = min((kcmax-kcb),(2-f)*kcmax);
    # ke=2.10;
    ke = kcmax - kcb
    pot_evapor = ke * et0

    # Potential Evapo-transpiration
    pot_et = pot_transpir + pot_evapor

    # Total available soil water that can be depleted from the root
    # zone before moisture stress starts
    depletable_water = p_tab + 0.04 * (5 - pot_et)
    dictionary = {"Tp": pot_transpir, "Ep": pot_evapor, "P": depletable_water, "f": frac_soil_cover}
    return dictionary


def getTotalDischarge(runoff, outlet_cells, drainage, baseflow=None):
    # m3
    if baseflow is None:
        tot_vol_disch_m3 = (runoff + outlet_cells + drainage)
    else:
        tot_vol_disch_m3 = (runoff + outlet_cells + drainage + baseflow)

    return tot_vol_disch_m3
