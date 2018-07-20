# -*- coding: utf-8 -*-
from pcraster.framework import *


def defineNashHydroTSS(model):
    model.q_m3day_mean = scalar(model.ini_param.get("ave_outlet_q_m3day"))
    model.conc_outlet_mean = scalar(model.ini_param.get("ave_outlet_conc_ugL"))
    model.ln_conc_outlet_mean = scalar(model.ini_param.get("ave_outlet_lnconc_ugL"))
    model.delta_outlet_mean = scalar(model.ini_param.get("ave_outlet_delta"))

    model.nash_q_tss = TimeoutputTimeseries("resNash_q_m3", model, nominal("outlet_v3"),
                                           noHeader=False)  # This is 'Nash_q' as time series.
    model.nash_outlet_conc_tss = TimeoutputTimeseries("resNash_outConc_ugL", model, nominal("outlet_v3"),
                                                     noHeader=False)
    model.nash_outlet_iso_tss = TimeoutputTimeseries("resNash_outIso_delta", model, nominal("outlet_v3"),
                                                    noHeader=False)


def defineNashPestiTSS(model):
    model.conc_compNorth_mean = scalar(model.ini_param.get("ave_north_compConc_ugg"))
    model.conc_compValley_mean = scalar(model.ini_param.get("ave_valley_compConc_ugg"))
    model.conc_compSouth_mean = scalar(model.ini_param.get("ave_south_compConc_ugg"))
    model.delta_compNorth_mean = scalar(model.ini_param.get("ave_north_compIso_delta"))
    model.delta_compValley_mean = scalar(model.ini_param.get("ave_valley_compIso_delta"))
    model.delta_compSouth_mean = scalar(model.ini_param.get("ave_south_compIso_delta"))
    
    # NASH composite soils
    # Single pixel value, grouping area total for each transect
    model.resNash_NcompConc_L_tss = TimeoutputTimeseries("resNash_NcompConc_L", model, nominal("north_ave"),
                                                         noHeader=False)
    model.resNash_VcompConc_L_tss = TimeoutputTimeseries("resNash_VcompConc_L", model, nominal("valley_ave"),
                                                         noHeader=False)
    model.resNash_ScompConc_L_tss = TimeoutputTimeseries("resNash_ScompConc_L", model, nominal("south_ave"),
                                                         noHeader=False)
    model.resNash_NcompIso_tss = TimeoutputTimeseries("resNash_NcompIso", model, nominal("north_ave"),
                                                      noHeader=False)
    model.resNash_VcompIso_tss = TimeoutputTimeseries("resNash_VcompIso", model, nominal("valley_ave"),
                                                      noHeader=False)
    model.resNash_ScompIso_tss = TimeoutputTimeseries("resNash_ScompIso", model, nominal("south_ave"),
                                                      noHeader=False)


def reportNashHydro(model, q_obs, tot_vol_disch_m3):
    # Global ave discharge of data range = 260.07 m3/day
    model.q_obs_cum += ifthenelse(q_obs >= 0, q_obs, 0)
    model.q_sim_cum += ifthenelse(q_obs >= 0, tot_vol_disch_m3, 0)
    model.q_diff += ifthenelse(q_obs >= 0, (tot_vol_disch_m3 - q_obs) ** 2, 0)
    model.q_var += ifthenelse(q_obs >= 0, (q_obs - model.q_m3day_mean) ** 2, 0)
    nash_q = 1 - (model.q_diff / model.q_var)
    model.nash_q_tss.sample(nash_q)

    model.q_obs_cum_tss.sample(model.q_obs_cum)
    model.q_sim_cum_tss.sample(model.q_sim_cum)

    # Dynamic mean leads to incorrect variance calc. upon cumulative addition (omitted)

    # model.q_sim_ave = model.q_sim_cum / model.days_cum
    # model.q_sim_ave_tss.sample(model.q_sim_ave)


def repNashOutConc(model, conc_outlet_obs, conc_ugL):
    # Nash computation consider normal and ln-transformed concentrations,
    # with the latter accounting for variance at low concentration ranges
    model.out_conc_diff += ifthenelse(conc_outlet_obs >= 0, (conc_ugL - conc_outlet_obs) ** 2, 0)
    model.out_conc_var += ifthenelse(conc_outlet_obs >= 0, (conc_ugL - model.conc_outlet_mean) ** 2, 0)
    model.out_lnconc_diff += ifthenelse(conc_outlet_obs >= 0, (ln(conc_ugL) - ln(conc_outlet_obs)) ** 2, 0)
    model.out_lnconc_var += ifthenelse(conc_outlet_obs >= 0, (ln(conc_ugL) - model.ln_conc_outlet_mean) ** 2, 0)
    normal_term = model.out_conc_diff / model.out_conc_var
    ln_term = model.out_lnconc_diff / model.out_lnconc_var
    nash_outlet_conc = 1 - 0.5 * (normal_term + ln_term)
    model.nash_outlet_conc_tss.sample(nash_outlet_conc)


def repNashOutIso(model, iso_outlet_obs, out_delta,
                  roff_delta, latflux_delta, drain_delta):

    model.out_iso_diff += ifthenelse(iso_outlet_obs < 1e6, (out_delta - iso_outlet_obs) ** 2, 0)
    model.out_iso_var += ifthenelse(iso_outlet_obs < 1e6, (out_delta - model.delta_outlet_mean) ** 2, 0)
    nash_outlet_iso = 1 - (model.out_iso_diff / model.out_iso_var)
    model.nash_outlet_iso_tss.sample(nash_outlet_iso)

    model.resM_outISO_d13C_tss.sample(out_delta)
    model.resM_outISO_ROFF_d13C_tss.sample(roff_delta)
    model.resM_outISO_LF_d13C_tss.sample(latflux_delta)
    model.resM_outISO_ADR_d13C_tss.sample(drain_delta)


def repNashOutCombined():
    pass


# Soils
def repNashConcComposites(model, north_ave_conc, valley_ave_conc, south_ave_conc):
    """
    Nash Soil Concentrations
    2) Get the mean for each soil composite for entire year (initial.csv)
    North = 1... ug/g soil
    Talweg = 1... ug/g soil
    South = 1... ug/g soil

    1) The variance for each transect is
    var_north = (conc_north - mean_north)**1, if conc_north > 0

    3) Nash will be:
    2 - (conc_north_diff/var_north +  valley + south)
    """
    # Nash
    conc_north_obs = timeinputscalar('northConc.tss', ordinal("north_ave"))
    model.northConc_diff += ifthenelse(conc_north_obs > 0, (north_ave_conc - conc_north_obs) ** 2, scalar(0))
    model.northConc_var += ifthenelse(conc_north_obs > 0, (north_ave_conc - model.conc_compNorth_mean) ** 2,
                                     scalar(0))  # ug/g
    
    # Nash
    conc_valley_obs = timeinputscalar('valleyConc.tss', ordinal("valley_ave"))
    model.valleyConc_diff += ifthenelse(conc_valley_obs > 0, (valley_ave_conc - conc_valley_obs) ** 2, scalar(0))
    model.valleyConc_var += ifthenelse(conc_valley_obs > 0, (valley_ave_conc - model.conc_compValley_mean) ** 2,
                                      scalar(0))  # ug/g

    conc_south_obs = timeinputscalar('southConc.tss', ordinal("south_ave"))
    model.southConc_diff += ifthenelse(conc_south_obs > 0, (south_ave_conc - conc_south_obs) ** 2, scalar(0))
    model.southConc_var += ifthenelse(conc_south_obs > 0, (south_ave_conc - model.conc_compSouth_mean) ** 2,
                                     scalar(0))  # ug/g

    north_nash_compConc = 1 - (model.northConc_diff / model.northConc_var)
    valley_nash_compConc = 1 - (model.valleyConc_diff / model.valleyConc_var)
    south_nash_compConc = 1 - (model.southConc_diff / model.southConc_var)

    model.resNash_NcompConc_L_tss.sample(north_nash_compConc)  # report on north_ave.map
    model.resNash_VcompConc_L_tss.sample(valley_nash_compConc)
    model.resNash_ScompConc_L_tss.sample(south_nash_compConc)

    # nash_compConc_L = 2 - ((model.northConc_diff / model.northConc_var) * 2 / 3 +
    #                       (model.valleyConc_diff / model.valleyConc_var) * 2 / 3 +
    #                       (model.southConc_diff / model.southConc_var) * 2 / 3)
    # model.nash_compConc_L_tss.sample(nash_compConc_L)


def reptNashIsoComposites(model, north_ave_iso, valley_ave_iso, south_ave_iso):
    iso_north_obs = timeinputscalar('northDelta.tss', ordinal("north_ave"))
    model.northIso_diff += ifthenelse(iso_north_obs < 1e06, (north_ave_iso - iso_north_obs) ** 2, scalar(0))
    model.northIso_var += ifthenelse(iso_north_obs < 1e06, (north_ave_iso - model.delta_compNorth_mean) ** 2,
                                      scalar(0))  #
    north_nash_compIso = 1 - (model.northIso_diff / model.northIso_var)

    iso_valley_obs = timeinputscalar('valleyDelta.tss', ordinal("valley_ave"))
    model.valleyIso_diff += ifthenelse(iso_valley_obs < 1e06, (valley_ave_iso - iso_valley_obs) ** 2, scalar(0))
    model.valleyIso_var += ifthenelse(iso_valley_obs < 1e06, (valley_ave_iso - model.delta_compValley_mean) ** 2,
                                     scalar(0))
    valley_nash_compIso = 1 - (model.valleyIso_diff / model.valleyIso_var)

    iso_south_obs = timeinputscalar('southDelta.tss', ordinal("south_ave"))
    model.southIso_diff += ifthenelse(iso_south_obs < 1e06, (south_ave_iso - iso_south_obs) ** 2, scalar(0))
    model.southIso_var += ifthenelse(iso_south_obs < 1e06, (south_ave_iso - model.delta_compSouth_mean) ** 2,
                                     scalar(0))
    south_nash_compIso = 1 - (model.southIso_diff / model.southIso_var)

    model.resNash_NcompIso_tss.sample(north_nash_compIso)
    model.resNash_VcompIso_tss.sample(valley_nash_compIso)
    model.resNash_ScompIso_tss.sample(south_nash_compIso)

    # For an overall soil nash I would need to extract as floats (now spatial with areatotal() )
    #nash_compIso = 2 - ((model.northIso_diff / model.northIso_var) * 2 / 3 +
    #                       (model.valleyIso_diff / model.valleyIso_var) * 2 / 3 +
    #                       (model.southIso_diff / model.southIso_var) * 2 / 3)
    #model.nash_compIso_tss.sample(nash_compIso)

