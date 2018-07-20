# -*- coding: utf-8 -*-
from pcraster.framework import *

"""
Nash Soil Concentrations
2) Get the mean for each soil composite for entire year
North = 2.909193 ug/g soil
Talweg = 1.261839 ug/g soil
South = 1.389668 ug/g soil

1) The variance for each transect is
var_north = (conc_north - mean_north)**1, if conc_north > 0

3) Nash will be:
2 - (conc_north_diff/var_north +  valley + south)
"""

# TODO: define codes for all transects...
north_plot_codes = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8']  # no 'n6'!
north_plots = len(north_plot_codes)


def importPlotMaps(model):
    # Points model to sampling points (pixels) on a given plot
    # Defines a reference to each map in dictionary: 'plot_maps'.
    try:
        model.plot_maps
    except AttributeError:
        model.plot_maps = dict()

    plots = ['north', 'n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8',
             'valley', 'v4', 'v5', 'v7', 'v8', 'v9', 'v10',
             'south', 's11', 's12', 's13']

    for plot in range(len(plots)):
        plot_map = plots[plot] + '_nom'
        model.plot_maps[plot_map] = nominal(model.readmap(plot_map))


def defineMassTSS(model, mMap="outlet_v3"):
    try:
        model.catch_dict
    except AttributeError:
        model.catch_dict = dict()

    names = [
        # Real
        "resM_light_real_z0",
        "resM_heavy_real_z0",
        "resM_light_real_zX",
        "resM_heavy_real_zX",
        # Aged
        "resM_light_aged_z0",
        "resM_heavy_aged_z0",
        "resM_light_aged_zX",
        "resM_heavy_aged_zX",
        # Transect Areas (only real)
        "resM_lhr_nor_z0",  # light + heavy (aged + bioav.)
        "resM_lhr_val_z0",
        "resM_lhr_sou_z0"
    ]
    for i in range(len(names)):
        tss_name = names[i]
        model.catch_dict[tss_name] = TimeoutputTimeseries(names[i], model, nominal(mMap), noHeader=False)


def reportSoilMass(model, name, var):
    model.catch_dict[name].sample(var)


def defineSoilTSS(model):
    try:
        model.soil_dict
    except AttributeError:
        model.soil_dict = dict()

    transects = ['north', 'valley', 'south']
    for tr in range(len(transects)):
        transect = transects[tr]
        transect_conc = transect[0:3] + 'CONC'
        transect_conc_real = transect[0:3] + 'CONC_real'
        transect_conc_aged = transect[0:3] + 'CONC_aged'
        transect_delta = transect[0:3] + 'd13C'
        transect_delta_real = transect[0:3] + 'd13C_real'
        transect_delta_aged = transect[0:3] + 'd13C_aged'
        transect_map = transect + '_ave'

        # Concentrations
        model.soil_dict[transect_conc] = TimeoutputTimeseries("resM_" + transect_conc, model, ordinal(transect_map),
                                                              noHeader=False)
        model.soil_dict[transect_conc_real] = TimeoutputTimeseries("resM_" + transect_conc_real, model,
                                                                   ordinal(transect_map), noHeader=False)
        model.soil_dict[transect_conc_aged] = TimeoutputTimeseries("resM_" + transect_conc_aged, model,
                                                                   ordinal(transect_map), noHeader=False)
        # Isotopes
        model.soil_dict[transect_delta] = TimeoutputTimeseries("resM_" + transect_delta, model, ordinal(transect_map),
                                                               noHeader=False)
        model.soil_dict[transect_delta_real] = TimeoutputTimeseries("resM_" + transect_delta_real, model,
                                                                    ordinal(transect_map), noHeader=False)
        model.soil_dict[transect_delta_aged] = TimeoutputTimeseries("resM_" + transect_delta_aged, model,
                                                                    ordinal(transect_map), noHeader=False)
    plots = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8',
             'v4', 'v5', 'v7', 'v8', 'v9', 'v10',
             's11', 's12', 's13']
    for plot in range(len(plots)):
        plot_name = plots[plot]
        plot_map = plot_name + '_out'  # 1-pixel map

        plot_conc = plot_name + 'CONC'
        plot_conc_real = plot_name + 'CONC_real'
        plot_conc_aged = plot_name + 'CONC_aged'

        plot_delta = plot_name + 'd13C'
        plot_delta_real = plot_name + 'd13C_real'
        plot_delta_aged = plot_name + 'd13C_aged'

        # Concentrations
        model.soil_dict[plot_conc] = TimeoutputTimeseries("resM_" + plot_conc, model, ordinal(plot_map), noHeader=False)
        model.soil_dict[plot_conc_real] = TimeoutputTimeseries("resM_" + plot_conc_real, model, ordinal(plot_map),
                                                               noHeader=False)
        model.soil_dict[plot_conc_aged] = TimeoutputTimeseries("resM_" + plot_conc_aged, model, ordinal(plot_map),
                                                               noHeader=False)

        # Delta
        model.soil_dict[plot_delta] = TimeoutputTimeseries("resM_" + plot_delta, model, ordinal(plot_map),
                                                           noHeader=False)
        model.soil_dict[plot_delta_real] = TimeoutputTimeseries("resM_" + plot_delta_real, model, ordinal(plot_map),
                                                                noHeader=False)
        model.soil_dict[plot_delta_aged] = TimeoutputTimeseries("resM_" + plot_delta_aged, model, ordinal(plot_map),
                                                                noHeader=False)
        # Example:
        # model.soil_dict[n1_d13C] = TimeoutputTimeseries("resM_n1d13C", model, ordinal("n1_out"), noHeader=False)


def reportSoilTSS(model, cell_mass, cell_massXdelta, transect, type='real'):
    if transect == 'north':
        plots = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8']
        plot_sampling_pts = [4, 6, 6, 4, 4, 5, 5]
        transect_sampling_pts = 30
    elif transect == 'valley':
        plots = ['v4', 'v5', 'v7', 'v8', 'v9', 'v10']
        plot_sampling_pts = [4, 4, 5, 5, 5, 5]
        transect_sampling_pts = 25
    else:
        transect = 'south'
        plots = ['s11', 's12', 's13']
        plot_sampling_pts = [8, 7, 5]
        transect_sampling_pts = 26

    # Record Transect
    transect_map = transect + '_nom'
    transect_tot_mass = areatotal(cell_mass, model.plot_maps[transect_map])
    transect_ave_mass = transect_tot_mass / scalar(transect_sampling_pts)
    transect_ave_conc = 1e6 * (transect_ave_mass /
                               (cellarea() * model.smp_depth)) * 1 / (model.p_bAgr * 1e03)  # ug/g soil
    transect_d13C = areatotal(cell_massXdelta, model.plot_maps[transect_map]) / transect_tot_mass

    if type == 'bioavail':
        transect_conc = transect[0:3] + 'CONC'
        transect_delta = transect[0:3] + 'd13C'
    elif type == 'real':
        transect_conc = transect[0:3] + 'CONC_real'
        transect_delta = transect[0:3] + 'd13C_real'
    else:
        transect_conc = transect[0:3] + 'CONC_aged'
        transect_delta = transect[0:3] + 'd13C_aged'

    model.soil_dict[transect_conc].sample(transect_ave_conc)
    model.soil_dict[transect_delta].sample(transect_d13C)

    # Record detailed
    assert len(plots) == len(plot_sampling_pts)
    for plot in range(len(plots)):
        plot_name = plots[plot]
        plot_map = plot_name + '_nom'
        tot_mass = areatotal(cell_mass, model.plot_maps[plot_map])
        plot_d13C = areatotal(cell_massXdelta, model.plot_maps[plot_map]) / tot_mass
        plot_ave_mass = tot_mass / scalar(plot_sampling_pts[plot])
        plot_ave_conc = 1e6 * (plot_ave_mass /
                               (cellarea() * model.smp_depth)) * 1 / (model.p_bAgr * 1e03)  # ug/g soil

        # Define dictionary key
        if type == 'bioavail':
            plot_conc = plot_name + 'CONC'
            plot_delta = plot_name + 'd13C'
        elif type == 'real':
            plot_conc = plot_name + 'CONC_real'
            plot_delta = plot_name + 'd13C_real'
        else:
            plot_conc = plot_name + 'CONC_aged'
            plot_delta = plot_name + 'd13C_aged'

        # Sample delta and concentrations
        model.soil_dict[plot_conc].sample(plot_ave_conc)
        model.soil_dict[plot_delta].sample(plot_d13C)

    # Report
    return {'ave_conc': transect_ave_conc,
            'd13C': transect_d13C}


def defineTransectSinkTSS(model, sink_name):
    try:
        model.sink_dict
    except AttributeError:
        model.sink_dict = dict()

    transects = ['north', 'valley', 'south']
    for tr in range(len(transects)):
        transect = transects[tr]
        transect_sink = transect[0:3] + sink_name
        transect_map = transect + '_ave'

        # Sinks (e.g. degradation or leaching or volat. )
        model.sink_dict[transect_sink] = TimeoutputTimeseries("resM_" + transect_sink, model, ordinal(transect_map),
                                                              noHeader=False)
    # plots = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8',
    #          'v4', 'v5', 'v7', 'v8', 'v9', 'v10',
    #          's11', 's12', 's13']
    # for plot in range(len(plots)):
    #     plot_name = plots[plot]
    #     plot_map = plot_name + '_out'  # 1-pixel map
    #
    #     plot_sink = plot_name + sink_name
    #
    #     # Sinks (e.g. degradation or leaching or volat. )
    #     model.sink_dict[plot_sink] = TimeoutputTimeseries("resM_" + plot_sink, model, ordinal(plot_map), noHeader=False)


def reportTransectSinkTSS(model, sink_name, sink, transect):

    if transect == 'north':
        plots = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8']
        plot_sampling_pts = [4, 6, 6, 4, 4, 5, 5]
        transect_sampling_pts = 30
    elif transect == 'valley':
        plots = ['v4', 'v5', 'v7', 'v8', 'v9', 'v10']
        plot_sampling_pts = [4, 4, 5, 5, 5, 5]
        transect_sampling_pts = 25
    else:
        transect = 'south'
        plots = ['s11', 's12', 's13']
        plot_sampling_pts = [8, 7, 5]
        transect_sampling_pts = 26

    # Record Transect
    transect_map = transect + '_nom'
    transect_tot_mass = areatotal(sink, model.plot_maps[transect_map])

    # Sink per sample (divide by 4 m2 to get per m2)
    transect_ave_mass = transect_tot_mass / scalar(transect_sampling_pts)

    transect_sink = transect[0:3] + sink_name

    model.sink_dict[transect_sink].sample(transect_ave_mass)

