# -*- coding: utf-8 -*-

# from time import *
# import os
# from copy import deepcopy
# import numpy as np
from datetime import datetime
# from pcraster._pcraster import *
# from pcraster.framework import *
from SALib.sample import latin

from mlhs_test import *

from applications import getApplications
from hydro_v2 import *
from nash import *
from pesti_v2 import *
from output_soils import *
from output import *
from test_suite import *


print(os.getcwd())

global state
state = -1


def get_state(old_state):
    new_state = old_state + 1
    global state
    state = new_state
    return state


def start_jday():
    start_sim = 166  # 213 # 166
    return start_sim


def getInputVector(row, sample_matrix, test=False):
    """
    :param row: relevant sample row
    :param sample_matrix: numpy sample matrix
    :return: a numpy row with required input parameters
    """
    if test:
        return sample_matrix
    test_vector = sample_matrix[row]
    return test_vector


class BeachModel(DynamicModel, MonteCarloModel):
    def setDebug(self):
        pass

    def __init__(self, cloneMap, names, params, upper, staticDT50=False, test=False):
        DynamicModel.__init__(self)
        MonteCarloModel.__init__(self)
        setclone(cloneMap)

        self.names = names  # Parameter names
        self.params = params  # Parameter matrix
        self.upper = upper  # Parameter upper bounds
        self.fixed_dt50 = staticDT50
        self.TEST = test

    def premcloop(self):
        self.DEBUG = False
        self.TEST_depth = False
        self.TEST_roots = False
        self.TEST_Ksat = False
        self.TEST_thProp = False
        self.TEST_theta = False
        self.TEST_IR = False
        self.TEST_PERC = False

        # Hydro
        self.LF = True
        self.ETP = True

        self.PEST = True
        self.TRANSPORT = True
        # Run fate processes
        self.ROM = True
        self.LCH = True
        self.ADRM = True
        self.LFM = True
        self.DEG = True

        self.TEST_LCH = False
        self.TEST_LFM = False
        self.TEST_DEG = False
        # This section includes all non-stochastic parameters.
        # Get initial parameters, make a dictionary of the raw file.
        import csv
        ini_path = 'initial.csv'
        self.ini_param = {}  # Dictionary to store the values
        with open(ini_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                self.ini_param[row[0].strip()] = float(row[1])

        date_path = 'Time.csv'
        self.time_dict = {}
        with open(date_path, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                self.time_dict[str(int(row[1]))] = str(row[0].strip())

        """
        Landscape Maps
        """
        self.dem = self.readmap("dem_slope")  # 192 - 231 m a.s.l
        self.datum_depth = (self.dem - mapminimum(self.dem)) * scalar(10 ** 3)  # mm

        # self.dem_route = self.readmap("dem_ldd")  # To route surface run-off
        # self.ldd_surf = lddcreate(self.dem_route, 1e31, 1e31, 1e31, 1e31)  # To route runoff
        out_burn = readmap("dem_ldd_burn3")
        self.is_catchment = defined(out_burn)
        self.is_north = readmap("norArea")
        self.is_valley = readmap("valArea")
        self.is_south = readmap("souArea")

        # self.ldd_subs = lddcreate(self.dem, 1e31, 1e31, 1e31, 1e31)  # To route lateral flow & build TWI
        self.ldd_subs = readmap('ldd_subs_v3')  # To route lateral flow & build TWI

        self.zero_map = out_burn - out_burn  # Zero map to generate scalar maps
        self.mask = out_burn / out_burn
        self.aging = deepcopy(self.zero_map)  # Cumulative days after application on each pixel

        self.outlet_multi = self.readmap("out_multi_nom_v3")  # Multi-outlet with 0 or 2
        self.is_outlet = boolean(self.outlet_multi == 1)

        importPlotMaps(self)

        self.landuse = self.readmap("landuse2016")

        # Topographical Wetness Index
        self.up_area = accuflux(self.ldd_subs, cellarea())
        self.slope = sin(atan(max(slope(self.dem), 0.001)))  # Slope in radians
        self.wetness = ln(self.up_area / tan(self.slope))

        """
        Output & Observations (tss and observation maps)
        """
        defineHydroTSS(self)  # output.py
        defineAverageMoistTSS(self)  # output.py

        definePestTSS(self)  # output.p
        defineSoilTSS(self)  # output_soils.py
        defineMassTSS(self)  # output_soils.py
        defineTransectSinkTSS(self, 'APP_mass')
        defineTransectSinkTSS(self, 'DEG_mass')
        defineTransectSinkTSS(self, 'AGE_mass')
        defineTransectSinkTSS(self, 'VOLA_mass')
        defineTransectSinkTSS(self, 'ROFF_mass')
        defineTransectSinkTSS(self, 'LCH_mass')
        #
        # defineNashHydroTSS(self)  # nash.py
        # defineNashPestiTSS(self)  # nash.py

    def initial(self):
        self.num_layers = int(self.ini_param.get("layers"))
        # Hydrological scenarios
        self.bsmntIsPermeable = False  # basement percolation (DP)
        self.ADLF = True
        self.bioavail = True

        # Morris_error tests
        m_state = get_state(state)  # First run will return state = 0

        vector = getInputVector(m_state, self.params, test=self.TEST)
        print("Vector " + str(m_state) + ": " + str(vector))

        z3_factor = self.mask * vector[self.names.index('z3_factor')] * self.upper[self.names.index('z3_factor')]

        """ Physical parameters for each layer """
        self.gamma = []  # coefficient to calibrate Ksat1
        self.c_lf = []
        for layer in range(self.num_layers):
            if layer < 2:
                # percolation coefficient
                self.gamma.append(self.mask * vector[self.names.index('gamma01')] * self.upper[self.names.index('gamma01')])
                # subsurface flow coefficient)
                self.c_lf.append(self.mask * vector[self.names.index('cZ0Z1')] * self.upper[self.names.index('cZ0Z1')])
            else:
                self.gamma.append(self.mask * vector[self.names.index('gammaZ')] * self.upper[self.names.index('gammaZ')])
                self.c_lf.append(self.mask * vector[self.names.index('cZ')] * self.upper[self.names.index('cZ')])

        self.c_adr = self.mask * vector[self.names.index('c_adr')] * self.upper[self.names.index('c_adr')]
        self.k_g = self.mask * vector[self.names.index('k_g')] * self.upper[self.names.index('k_g')]  # [days]
        self.gw_factor = 1 - z3_factor  # Represents bottom-most portion of bottom layer
        self.f_transp = self.mask * vector[self.names.index("f_transp")] * self.upper[self.names.index('f_transp')]  # [-]

        self.drainage_layers = [False, False, True, False, False]  # z2 turned on!!

        """
        Hydro Maps
        """

        # theta_sat_z2 = self.zero_map + readmap("thSATz2")  # 0.63  scalar(self.ini_param.get("sat_z2z3"))
        # # scalar(self.ini_param.get("sat_z2z3")) + mapnormal() * 0.04  # mean + 1SD(X)*0.04 = mean + (0.002)**0.5
        # theta_fcap_z2 = self.zero_map + readmap("thFCz2") # scalar(self.ini_param.get("fc_z2z3")) # => 0.38702
        # scalar(self.ini_param.get("fc_z2z3")) + mapnormal() * 0.04  # mean + 1SD(X)*0.04 = mean + (0.002)**0.5

        # Initial moisture (Final from model v1, Sept 30, 2016)
        self.theta = []
        self.theta_sat = []
        self.theta_fc = []
        # self.theta_100 = []
        self.theta_wp = []  # => 0.19
        for layer in range(self.num_layers):
            if layer < 2:
                self.theta_sat.append(deepcopy(self.zero_map))
                self.theta_fc.append(deepcopy(self.zero_map))
                self.theta_wp.append(self.mask * scalar(self.ini_param.get("WPZ01")))
            elif layer == 2:
                self.theta_sat.append(self.mask * scalar(self.ini_param.get("SATZ")))
                self.theta_fc.append(self.mask * scalar(self.ini_param.get("FCZ")))
                self.theta_wp.append(self.mask * scalar(self.ini_param.get("WPZ")))
            else:
                self.theta_sat.append(self.mask * scalar(self.ini_param.get("SATZ")))
                self.theta_fc.append(self.mask * scalar(self.ini_param.get("FCZ")))
                self.theta_wp.append(self.mask * scalar(self.ini_param.get("WPZ")))

            if start_jday() < 100:
                name = 'd14_theta_z' + str(layer)
                self.theta.append(readmap(name))
            else:
                name = 'd166_theta_z' + str(layer)
                self.theta.append(readmap(name))

        """ Soil Properties """
        # Soil bulk density (g/cm^3)
        self.p_bZ = scalar(self.ini_param.get("p_bZ"))
        # Organic Carbon in soil without grass (kg/kg)
        self.f_oc = self.mask * vector[self.names.index('f_oc')] * self.upper[self.names.index('f_oc')]

        """
        Sorption parameters
        """
        # K_oc - S-metolachlor (K_oc in ml/g)
        # Marie's thesis: log(K_oc)= 2.8-1.6 [-] -> k_oc = 63 - 398 (Alletto et al., 2013).
        # Pesticide Properties Database: k_oc = 120 ml/g (range: 50-540 mL/g)
        self.k_oc = self.mask * vector[self.names.index('k_oc')] * self.upper[self.names.index('k_oc')]  # ml/g
        self.k_d = self.k_oc * self.f_oc  # Dissociation coefficient K_d (mL/g = L/Kg)

        # Pesticide Properties Database states :
        # K_d=0.67; but here
        # K_d=120*0.021 = 1.52;  (L/kg)
        # Difference will lead to higher retardation factor (more sorption)
        """ Runoff transport """
        self.beta_runoff = self.mask * \
                           vector[self.names.index('beta_runoff')] * self.upper[self.names.index('beta_runoff')]  # mm

        """
        Volatilization parameters
        """
        # Henry's constant @ 20 C (Metolachlor, Feigenbrugel et al., 2004)
        self.k_cp = scalar(self.ini_param.get("k_cp"))  # mol/L atm
        # Henry, dimensionless conversion Hcc = Hcp*R*T
        self.k_h = self.k_cp * 0.0821 * 273.15  # scalar(self.ini_param.get("k_h"))
        self.molar = scalar(self.ini_param.get("molar"))  # g S-met/mol

        """
        Degradation parameters
        """
        self.temp_ref = scalar(self.ini_param.get("temp_ref"))  # Temp.  reference
        self.theta_ref = scalar(self.ini_param.get("theta_ref"))  # Theta  reference
        self.act_e = scalar(self.ini_param.get("activation_e"))  # Metolachlor Ea = 23.91 KJ/mol; @Jaikaew2017
        self.r_gas = scalar(self.ini_param.get("r_gas"))  # R = 8.314 J / mol Kelvin,

        """
        Isotopes
        """
        self.r_standard = scalar(self.ini_param.get("r_standard"))  # VPDB
        epsilon_iso = -self.mask * vector[self.names.index('epsilon_iso')] * self.upper[self.names.index('epsilon_iso')]
        self.alpha_iso = epsilon_iso / 1000 + 1

        """
        Degradation
        """
        self.dt_50_ref = self.mask * vector[self.names.index('dt_50_ref')] * self.upper[self.names.index('dt_50_ref')]  # S-met (days)
        self.dt_50_aged = self.mask * vector[self.names.index('dt_50_aged')] * self.upper[self.names.index('dt_50_aged')] # Ageing rate days
        self.dt_50_ab = self.mask * vector[self.names.index('dt_50_ab')] * self.upper[self.names.index('dt_50_ab')] # Abiotic degradation rate 1/d
        self.beta_moisture = self.mask * vector[self.names.index('beta_moisture')] * self.upper[self.names.index('beta_moisture')]
        """
        Layer depths
        """

        self.layer_depth = []
        self.tot_depth = deepcopy(self.zero_map)
        bottom_depth = deepcopy(self.zero_map)
        for layer in range(self.num_layers):
            if layer < self.num_layers - 2:  # 5 - 1 = 3 (i.e. z0,z1,z2)
                self.layer_depth.append(self.zero_map +
                                        scalar(self.ini_param.get('z' + str(layer))))
                self.tot_depth += self.layer_depth[layer]
                # self.report(self.layer_depth[layer], 'DepthZ' + str(layer))
            elif layer < self.num_layers - 1:  # 5 - 2 = 4 (i.e. z3)
                bottom_depth = (self.datum_depth +  # total height
                                scalar(self.ini_param.get('z' + str(layer))) + 100  # plus a min-depth
                                - self.tot_depth)
                self.layer_depth.append(bottom_depth * z3_factor)  # minus:(z0, z1, z2)*decreasing depth factor
                self.tot_depth += self.layer_depth[layer]
                # self.report(self.layer_depth[layer], 'DepthZ' + str(layer))
            else:  # Basement Layer = n5  (z4)
                self.layer_depth.append(bottom_depth * self.gw_factor)  # minus:(z0, z1, ...)*decreasing depth factor
                self.tot_depth += self.layer_depth[layer]
                # self.report(self.layer_depth[layer], 'DepthZ' + str(layer))

            if self.TEST_depth:
                checkLayerDepths(self, layer)

        self.smp_depth = self.layer_depth[0]
        # self.report(self.tot_depth, 'zTot_mm')
        #  aguila --scenarios='{2}' DepthZ0 DepthZ1 DepthZ2 DepthZ3 DepthZ4 zTot_mm

        """
        Pesticides Maps
        """
        # Application days
        self.app_days = [171, 177, 196, 200, 213, 238, 245]
        self.aged_days = ifthen(boolean(self.is_catchment), scalar(365))

        # Mass
        # in ug = conc. (ug/g soil) * density (g/cm3) * (10^6 cm3/m3)*(2 m/10^3 mm)* depth_layer(mm) * cellarea(m2)
        # g = ug * 1e-06
        self.sm_background = []
        mean_back_conc = [0.06, 0.03, 0.001, 0.001, 0.001]
        for layer in range(self.num_layers):
            background = ((self.zero_map + mean_back_conc[layer]) * self.p_bZ * scalar(10 ** 6 / 10 ** 3) *
                          self.layer_depth[layer] * cellarea() * (10 ** -6))  # Based on detailed soils
            self.sm_background.append(background)

        # Fraction masses and Delta (Background)
        self.lightmass = []
        self.lightmass_ini = []
        self.lightaged_ini = []
        self.heavymass = []
        self.heavymass_ini = []
        self.heavyaged_ini = []

        self.delta = []
        self.delta_ini = []
        self.delta_real = []
        self.delta_aged = []
        self.light_back = []
        self.heavy_back = []

        self.light_aged = []
        self.heavy_aged = []

        self.light_real = []
        self.heavy_real = []

        # Fraction bioavailable after 365 days (with Ageing model Mt = M0 * exp(-0.005*time); ct/C0 (time = 365) = 0.16
        avail_frac = 0.01
        aged_frac = 1 - avail_frac

        # Initial Isotope Signature
        for layer in range(self.num_layers):
            # Initial deltas assume theoretical max @99% deg Streitwieser Semiclassical Limits
            self.delta.append(self.zero_map - 23.7)
            self.delta_ini.append(self.zero_map - 23.7)
            self.delta_real.append(self.zero_map - 23.7)
            self.delta_aged.append(self.zero_map - 23.7)

            # Aged fraction, still available for biodegradation
            self.light_back.append(self.sm_background[layer] * avail_frac /
                                   (1 + self.r_standard * (self.delta[layer] / 1000 + 1)))
            self.heavy_back.append(self.sm_background[layer] * avail_frac - self.light_back[layer])

            # Aged fraction - not available for biodegradation
            self.light_aged.append(self.sm_background[layer] * aged_frac /
                                   (1 + self.r_standard * (self.delta_aged[layer] / 1000 + 1)))
            self.heavy_aged.append(self.sm_background[layer] * aged_frac - self.light_aged[layer])

            # Set mass fractions <- background fractions
            self.lightmass.append(deepcopy(self.light_back[layer]))
            self.lightmass_ini.append(deepcopy(self.light_back[layer]))

            self.heavymass.append(deepcopy(self.heavy_back[layer]))
            self.heavymass_ini.append(deepcopy(self.heavy_back[layer]))

            self.lightaged_ini.append(deepcopy(self.light_aged[layer]))
            self.heavyaged_ini.append(deepcopy(self.heavy_aged[layer]))

            # Combined bioavailable and aged masses
            self.light_real.append(self.lightmass[layer] + self.light_aged[layer])
            self.heavy_real.append(self.heavymass[layer] + self.heavy_aged[layer])

            if mapminimum(self.lightmass[layer]) < 0:
                print("Err INI, light")

            if mapminimum(self.heavymass[layer]) < 0:
                print("Err INI, heavy")

        # Assign dosages based on Farmer-Crop combinations [g/m2]
        self.fa_cr = readmap("farm_burn_v3")  # Contains codes to assign appropriate dosage
        self.plot_codes = readmap("plot_code16")  # Contains codes to assign appropriate dosage
        self.apps = getApplications(self, self.fa_cr, self.plot_codes, massunit='g')  # returns list of applied masses

        # Applications delta
        # Use map algebra to produce a initial signature map,
        # ATT: Need to do mass balance on addition of new layer.
        # where app1 > 0, else background sig. (plots with no new mass will be 0)
        # where app1 > 0, else background sig. (plots with no new mass will be 0)
        self.appDelta = []
        for a in range(len(self.apps)):
            self.appDelta.append(ifthenelse(self.apps[a] > 0, scalar(-32.3), scalar(-23.7)))

        # Cumulative maps
        self.cum_runoff_ug = deepcopy(self.zero_map)
        self.cum_leached_ug_z0 = deepcopy(self.zero_map)
        self.cum_leached_ug_z1 = deepcopy(self.zero_map)
        self.cum_leached_ug_z2 = deepcopy(self.zero_map)
        self.cum_leached_ug_z3 = deepcopy(self.zero_map)

        self.cum_latflux_ug_z0 = deepcopy(self.zero_map)
        self.cum_latflux_ug_z1 = deepcopy(self.zero_map)
        self.cum_latflux_ug_z2 = deepcopy(self.zero_map)
        self.cum_latflux_ug_z3 = deepcopy(self.zero_map)

        self.cum_baseflx_ug_z3 = deepcopy(self.zero_map)

        self.cum_appZ0_g = deepcopy(self.zero_map)
        self.cum_degZ0_g = deepcopy(self.zero_map)
        self.cum_aged_deg_L_g = deepcopy(self.zero_map)
        self.cum_deg_L_g = deepcopy(self.zero_map)  # Total cum deg
        self.cum_roZ0_L_g = deepcopy(self.zero_map)
        self.cum_volatZ0_L_g = deepcopy(self.zero_map)
        self.cum_lchZ0_L_g = deepcopy(self.zero_map)
        self.cum_adr_L_g = deepcopy(self.zero_map)
        self.cum_latflux_L_g = deepcopy(self.zero_map)
        self.cum_exp_L_g = deepcopy(self.zero_map)
        self.northConc_diff = deepcopy(self.zero_map)
        self.northConc_var = deepcopy(self.zero_map)
        self.valleyConc_diff = deepcopy(self.zero_map)
        self.valleyConc_var = deepcopy(self.zero_map)
        self.southConc_diff = deepcopy(self.zero_map)
        self.southConc_var = deepcopy(self.zero_map)
        self.northIso_diff = deepcopy(self.zero_map)
        self.northIso_var = deepcopy(self.zero_map)
        self.valleyIso_diff = deepcopy(self.zero_map)
        self.valleyIso_var = deepcopy(self.zero_map)
        self.southIso_diff = deepcopy(self.zero_map)
        self.southIso_var = deepcopy(self.zero_map)

        """
        Temperature maps and params
        """
        self.lag = scalar(0.8)  # lag coefficient (-), 0 < lag < 2; -> in SWAT, lag = 0.80
        # Generating initial surface temp map (15 deg is arbitrary)
        self.temp_fin = []
        self.temp_surf_fin = self.zero_map + 15
        for layer in range(self.num_layers):
            self.temp_fin.append(self.zero_map + 15)

        # Maximum damping depth (dd_max)
        # The damping depth (dd) is calculated daily and is a function of max. damping depth (dd_max), (mm):
        self.dd_max = (scalar(2500) * self.p_bZ) / (self.p_bZ + 686 * exp(-5.63 * self.p_bZ))

        # TODO
        # Average Annual air temperature (celcius - Layon!! Not Alteckendorf yet!!)
        self.temp_ave_air = scalar(12.2)  # 12.1 is for Layon

        """
        Simulation start time
        """
        start_day = start_jday()  # Returns initial timestep
        greg_date = self.time_dict[str(start_day)].split("/", 2)
        print("Date: ", greg_date[0], greg_date[1], greg_date[2])
        print("Sim Day: ", start_day)
        yy = int(greg_date[2])
        mm = int(greg_date[1])
        dd = int(greg_date[0])

        date_factor = 1
        if (100 * yy + mm - 190002.5) < 0:
            date_factor = -1

        # simulation start time in JD (Julian Day)
        self.jd_start = 367 * yy - rounddown(7 * (yy + rounddown((mm + 9) / 12)) / 4) + rounddown(
            (275 * mm) / 9) + dd + 1721013.5 - 0.5 * date_factor
        self.jd_cum = 0
        self.jd_dt = 1  # Time step size (days)

        # Analysis
        self.water_balance = []  # mm
        for layer in range(self.num_layers):
            self.water_balance.append(deepcopy(self.zero_map))

        # Nash discharge
        self.days_cum = 0  # Track no. of days with data
        self.q_diff = 0
        self.q_var = 0
        self.q_obs_cum = 0
        self.q_sim_cum = 0  # net total disch
        self.q_sim_ave = 0

        # Nash concentration outlet
        self.out_conc_diff = 0
        self.out_conc_var = 0
        self.out_lnconc_diff = 0
        self.out_lnconc_var = 0

        # Nash isotopes outlet
        self.out_iso_diff = 0
        self.out_iso_var = 0

        self.rain_cum_m3 = 0  # Rainfall
        self.rain_cum_mm = self.zero_map + scalar(400.0)  # Cum Rainfall

        self.tot_drain_m3 = 0  # drainage z1
        # self.tot_nlf_m3 = 0
        # self.tot_ilf_m3 = 0  # upstream inflow
        self.cum_olf_m3 = 0  # downstream outflow

        self.tot_of_m3 = 0  # Overflow due to LF sat capacity reached

        self.tot_etp_m3 = 0
        self.tot_baseflow_m3 = 0
        self.tot_perc_z3_m3 = 0
        self.tot_runoff_m3 = 0

        # Stochastic / test parameters
        print("state:", m_state)

        # Need initial states to compute change in storage after each run
        self.theta_ini = deepcopy(self.theta)

    def dynamic(self):

        jd_sim = self.jd_start + self.jd_cum
        if self.PEST:
            self.aged_days += scalar(1)
        # timeinputscalar() gets the TSS's cell value of row (timestep) and TSS's column indexed by the landuse-map.
        # In other words, the value of the landuse-map pixel == column to to look for in landuse.tss
        # So currently becasue landuse does not change value in the year, this step is redundant
        # and we could simply use the landuse map to map the fields to the "Crop Parameters" below.
        # Mapping "landuse.map" to -> "fields map" (i.e. the latter is a dyanmic-landuse equivalent).
        fields = timeinputscalar('landuse.tss', nominal(self.landuse))  #
        # Note that the number of columns could still be reduced to 9 as, only 9 classes are considered in 2016.

        " Crop Parameters "
        # SEE: http://pcraster.geo.uu.nl/pcraster/4.1.0/doc/manual/op_lookup.html?highlight=lookupscalar
        setglobaloption('matrixtable')  # allows lookupscalar to read more than 1 expressions.
        crop_type = lookupscalar('croptable.tbl', 1, fields)  # (table, col-value in crop table, column-value in fields)
        sow_yy = lookupscalar('croptable.tbl', 2, fields)
        sow_mm = lookupscalar('croptable.tbl', 3, fields)  # sowing or Greenup month
        sow_dd = lookupscalar('croptable.tbl', 4, fields)  # sowing day
        sow_dd = ifthenelse(self.fa_cr == 1111, sow_dd - 15,  # Beet Friess
                            sow_dd)
        len_grow_stage_ini = lookupscalar('croptable.tbl', 5,
                                          fields)  # old: Lini. length of initial crop growth stage
        len_dev_stage = lookupscalar('croptable.tbl', 6, fields)  # Ldev: length of development stage
        len_mid_stage = lookupscalar('croptable.tbl', 7, fields)  # Lmid: length of mid-season stage
        len_end_stage = lookupscalar('croptable.tbl', 8, fields)  # Lend: length of late season stage
        kcb_ini = lookupscalar('croptable.tbl', 9, fields)  # basal crop coefficient at initial stage
        kcb_mid = lookupscalar('croptable.tbl', 10, fields)  # basal crop coefficient at mid season stage
        kcb_end = lookupscalar('croptable.tbl', 11, fields)  # basal crop coefficient at late season stage
        max_LAI = lookupscalar('croptable.tbl', 12, fields)  # maximum leaf area index
        mu = lookupscalar('croptable.tbl', 13, fields)  # light use efficiency
        max_height = lookupscalar('croptable.tbl', 14, fields)  # maximum crop height

        max_root_depth = lookupscalar('croptable.tbl', 15, fields) * 1000  # max root depth converting from m to mm
        min_root_depth = scalar(150)  # Seeding depth [mm], Allen advices: 0.15 to 0.20 m

        # Max RD (m) according to Allen 1998, Table 22 (now using FAO source)
        # Sugar beet = 0.7 - 2.1
        # Corn = 2.0 - 2.7
        # Grazing pasture 0.5 - 2.5
        # Spring Wheat = 2.0 -2.5
        # Winter Wheat = 2.5 -2.8
        # Apple trees = 2.0-1.0

        # depletable theta before water stress (Allen1998, Table no.22)
        p_tab = lookupscalar('croptable.tbl', 16, fields)
        # Sugar beet = 0.55
        # Corn = 0.55
        # Grazing Pasture = 0.6
        # Spring Wheat = 0.55
        # Winter Wheat = 0.55
        # Apple trees = 0.5

        """ Soil physical parameters
        """
        # Basement layers defined under initial()
        self.theta_sat[0] = timeinputscalar('thetaSat_agr.tss', nominal(self.landuse))  # saturated moisture # [-]
        self.theta_sat[1] = deepcopy(self.theta_sat[0])
        self.theta_fc[0] = timeinputscalar('thetaFC_agr.tss', nominal(self.landuse))  # * self.fc_adj  # field capacity
        self.theta_fc[1] = deepcopy(self.theta_fc[0])

        # print(self.currentTimeStep())
        excess_z0 = ifthenelse(self.theta[0] > self.theta_sat[0], self.theta[0] - self.theta_sat[0], scalar(0))
        excess_z1 = ifthenelse(self.theta[1] > self.theta_sat[1], self.theta[1] - self.theta_sat[1], scalar(0))
        self.theta[0] = ifthenelse(self.theta[0] > self.theta_sat[0], self.theta_sat[0], self.theta[0])
        self.theta[1] = ifthenelse(self.theta[1] > self.theta_sat[1], self.theta_sat[1], self.theta[1])

        if self.TEST_thProp:
            checkMoistureProps(self, self.theta_sat, 'aSATz')
            checkMoistureProps(self, self.theta_fc, 'aFCz')

        self.p_bAgr = timeinputscalar('p_b_agr.tss', nominal(self.landuse))
        k_sat_z0z1 = timeinputscalar('ksats.tss', nominal(self.landuse))
        k_sat_z2z3 = lookupscalar('croptable.tbl', 17, fields)  # saturated conductivity of the second layer
        k_sat = []
        for i in range(self.num_layers):
            if i < 2:
                k_sat.append(deepcopy(k_sat_z0z1))
            else:
                k_sat.append(deepcopy(k_sat_z2z3))

        CN2_A = lookupscalar('croptable.tbl', 18, fields)  # curve number of moisture condition II
        CN2_B = lookupscalar('croptable.tbl', 19, fields)  # curve number of moisture condition II
        CN2_C = lookupscalar('croptable.tbl', 20, fields)  # curve number of moisture condition II
        CN2_D = lookupscalar('croptable.tbl', 21, fields)  # curve number of moisture condition II

        if self.TEST_Ksat:
            reportKsatEvolution(self, k_sat)

        """
        Time-series data to spatial location,
        map is implicitly defined as the clonemap.
        """
        precip = timeinputscalar('rain.tss', 1)  # daily precipitation data as time series (mm)
        # Precipitation total
        rain_m3 = self.mask * precip * cellarea() / 1000  # m3
        tot_rain_m3 = areatotal(rain_m3, self.is_catchment)

        temp_bare_soil = timeinputscalar('T_bare.tss', nominal('clone_nom'))  # SWAT, Neitsch2009, p.43.
        self.temp_air = timeinputscalar('airTemp.tss', nominal('clone_nom'))
        et0 = timeinputscalar('ET0.tss', 1)  # daily ref. ETP at Zorn station (mm)
        wind = timeinputscalar('U2.tss', 1)  # wind speed time-series at 1 meters height
        humid = timeinputscalar('RHmin.tss', 1)  # minimum relative humidity time-series # PA: (-)
        # precipVol = precip * cellarea() / 1000  # m3

        ################
        # Crop growth ##
        ################
        jd_sow = convertJulian(sow_yy, sow_mm, sow_dd)
        self.rain_cum_mm += precip
        self.rain_cum_mm = ifthenelse(jd_sim == jd_sow, scalar(0), self.rain_cum_mm)
        # self.report(self.rain_cum_mm, 'aCuRain')
        CN2 = ifthenelse(self.rain_cum_mm > 60, CN2_D, CN2_A)  # report_CN(self, CN2, self.rain_cum_mm)

        # soil_group = ifthenelse(self.rain_cum_mm > 90, ordinal(3), ordinal(2))
        all_stages = len_grow_stage_ini + len_dev_stage + len_mid_stage + len_end_stage

        # updating of sowing date by land use
        # TODO: Why update if sow date is already by land use?
        # sow_yy = ifthenelse(jd_sim < jd_sow + all_stages, sow_yy,
        #                     ifthenelse(jd_sim < jd_sow + all_stages + 365, sow_yy + 2,
        #                                ifthenelse(jd_sim < jd_sow + all_stages + 730, sow_yy + 1,
        #                                           ifthenelse(jd_sim < jd_sow + all_stages + 1095, sow_yy + 3,
        #                                                      ifthenelse(jd_sim < jd_sow + all_stages + 1460,
        #                                                                 sow_yy + 4,
        #                                                                 scalar(0))))))

        # Update sowing date / plant date
        jd_plant = convertJulian(sow_yy, sow_mm, sow_dd)

        jd_dev = jd_plant + len_grow_stage_ini
        jd_mid = jd_dev + len_dev_stage
        jd_late = jd_mid + len_mid_stage
        jd_end = jd_late + len_end_stage
        LAIful = max_LAI + 0.5

        height = timeinputscalar('height.tss', nominal(self.landuse))
        # root_depth_tot2 = timeinputscalar('height.tss', nominal(self.landuse)) * self.root_adj
        # root_depth_tot2 *= 10 ** 3  # Convert to mm

        root_depth_tot = ifthenelse(jd_sim > jd_mid, max_root_depth,  # Passed development stage
                                    ifthenelse(jd_sim > jd_plant,  # Planted, but before attaining max root depth
                                               min_root_depth + ((max_root_depth - min_root_depth) *
                                                                 ((jd_sim - jd_plant) / (jd_mid - jd_plant))),
                                               scalar(0)))  # Before planting
        # self.report(root_depth_tot, 'RDtot1')

        root_depth = []
        for layer in range(self.num_layers):
            if layer == 0:
                root_depth_z0 = ifthenelse(root_depth_tot > self.layer_depth[0], self.layer_depth[0], root_depth_tot)
                root_depth.append(root_depth_z0)
            elif layer == 1:
                root_depth_z1 = ifthenelse(root_depth_tot < self.layer_depth[0], scalar(0),
                                           ifthenelse(root_depth_tot <= self.layer_depth[1] + self.layer_depth[0],
                                                      root_depth_tot - self.layer_depth[0], self.layer_depth[1]))
                root_depth.append(root_depth_z1)
            elif layer == 2:
                root_depth_z2 = ifthenelse(root_depth_tot <= self.layer_depth[0] + self.layer_depth[1], scalar(0),
                                           ifthenelse(
                                               root_depth_tot <= self.layer_depth[0] + self.layer_depth[1] +
                                               self.layer_depth[2],
                                               root_depth_tot - self.layer_depth[0] - self.layer_depth[1],
                                               self.layer_depth[2]))
                root_depth.append(root_depth_z2)
            elif layer == 3:
                root_depth_z3 = ifthenelse(
                    root_depth_tot <= self.layer_depth[0] + self.layer_depth[1] + self.layer_depth[2],
                    scalar(0),
                    ifthenelse(
                        root_depth_tot <= self.layer_depth[0] + self.layer_depth[1] + self.layer_depth[2] +
                        self.layer_depth[3],
                        root_depth_tot - self.layer_depth[0] - self.layer_depth[1] - self.layer_depth[2],
                        self.layer_depth[3]))
                root_depth.append(root_depth_z3)
            else:
                root_depth.append(scalar(0))

        if self.TEST_roots:
            checkRootDepths(self, root_depth)


        # calculation of fraction of soil covered by vegetation
        # frac_soil_cover = 2 - exp(-mu * LAI)
        # \mu is a light-use efficiency parameter that
        # depends on land-use characteristics
        # (i.e. Grass: 0.35; Crops: 0.45; Trees: 0.5-0.77; cite: Larcher, 1975).

        # TODO: Check "f" definition by Allan et al., 1998 against previous (above)
        # fraction of soil cover is calculated inside the "getPotET" function.
        # frac_soil_cover = ((Kcb - Kcmin)/(Kcmax - Kcmin))**(2+0.5*mean_height)
        # self.fTss.sample(frac_soil_cover)

        # Get potential evapotranspiration for all layers
        etp_dict = getPotET(self, sow_yy, sow_mm, sow_dd, root_depth_tot, min_root_depth,
                            jd_sim,
                            wind, humid,
                            et0,
                            kcb_ini, kcb_mid, kcb_end,
                            height,
                            len_grow_stage_ini, len_dev_stage, len_mid_stage, len_end_stage,
                            p_tab)
        pot_transpir = etp_dict["Tp"]
        pot_evapor = etp_dict["Ep"]
        depletable_water = etp_dict["P"]
        # TODO: printouts!
        # self.report(pot_transpir, 'aPotTRA')
        # self.report(pot_evapor, 'aPotEVA')

        # Not in use for water balance, but used to estimate surface temp due to bio-cover.
        frac_soil_cover = etp_dict["f"]
        # self.report(frac_soil_cover, 'aFracCV')

        bio_cover = getBiomassCover(self, frac_soil_cover)
        # bcv should range 0 (bare soil) to 2 (complete cover)
        # self.report(bio_cover, 'aBCV')

        # Applications
        mass_applied = deepcopy(self.zero_map)
        light_applied = deepcopy(self.zero_map)
        heavy_applied = deepcopy(self.zero_map)
        light_volat = deepcopy(self.zero_map)
        heavy_volat = deepcopy(self.zero_map)

        """
        Applications and Volatilization (on application days only)
        """
        if self.currentTimeStep() in self.app_days:
            indx = self.app_days.index(self.currentTimeStep())
            mass_applied = ifthenelse(self.currentTimeStep() == self.app_days[indx], self.apps[indx], scalar(0))
            light_applied = ifthenelse(self.currentTimeStep() == self.app_days[indx],
                                       getLightMass(self, mass_applied, indx), scalar(0))
            heavy_applied = mass_applied - light_applied
            self.lightmass[0] += light_applied
            self.heavymass[0] += heavy_applied
            self.cum_appZ0_g += light_applied + heavy_applied
            self.aged_days = ifthenelse(mass_applied > 0, scalar(0), self.aged_days)

            if mapminimum(self.lightmass[0]) < 0:
                print("Err APP, light")

            if mapminimum(self.heavymass[0]) < 0:
                print("Err APP, heavy")

            if self.TRANSPORT:
                # Mass volatilized
                layer = 0
                light_volat = getVolatileMass(self, self.temp_air, self.lightmass[layer],  # "LL",
                                              rel_diff_model='option-2', sorption_model="linear",
                                              gas=True, run=self.PEST)
                heavy_volat = getVolatileMass(self, self.temp_air, self.heavymass[layer],  # "HH",
                                              rel_diff_model='option-2', sorption_model="linear",
                                              gas=True, run=self.PEST)

                light_volat = ifthenelse(mass_applied > 0, light_volat, scalar(0))
                heavy_volat = ifthenelse(mass_applied > 0, heavy_volat, scalar(0))

                self.lightmass[layer] -= light_volat
                self.heavymass[layer] -= heavy_volat
                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err Volat, light")
                    self.report(self.lightmass[layer], 'VOL_Mz0')

                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err Volat, heavy")

        """
        Infiltration, runoff, & percolation (all layers)
        """
        infil_z0 = deepcopy(self.zero_map)
        runoff_z0 = deepcopy(self.zero_map)
        percolation = []
        mass_runoff = []  # 0 <- light, 2 <- heavy
        light_leached = []
        heavy_leached = []
        permeable = True
        for layer in range(self.num_layers):
            if layer == 0:  # Layer 0
                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err Start, light")
                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err Start, heavy")

                # Excess due to changes in saturation capacities
                precip += (excess_z0 + excess_z1)  # One approach to distribute excess moisture
                z0_IRO = getTopLayerInfil(self, precip, CN2, crop_type,
                                          jd_sim, jd_dev, jd_mid, jd_end, len_dev_stage)
                runoff_z0 = z0_IRO.get("roff")  # [mm]
                # Partition infiltration
                infil_z0 = z0_IRO.get("infil_z0")  # [mm]
                infil_z1 = z0_IRO.get("infil_z1")  # [mm]

                # Distribute to layer 0
                SW0 = self.theta[layer] * self.layer_depth[layer] + infil_z0
                self.theta[layer] = SW0 / self.layer_depth[layer]  # [-]

                # Distribution to layer z1 needed here, bc. getPercolation(layer = 0) will check below's capacity
                SW1 = self.theta[layer + 1] * self.layer_depth[layer + 1] + infil_z1
                self.theta[layer + 1] = SW1 / self.layer_depth[layer + 1]  # [-]

                excess = max(self.theta[layer] - self.theta_sat[layer], scalar(0))
                if mapmaximum(excess) > 0:
                    val = float(mapmaximum(excess))
                    self.theta[layer] = ifthenelse(self.theta[layer] > self.theta_sat[layer], self.theta_sat[layer],
                                                   self.theta[layer])
                    if float(val) > float(1e-02):
                        print("Corrected Percolation(), SAT was exceeded, layer " + str(layer) + ' by ' + str(val))

                excessLj = max(self.theta[layer + 1] - self.theta_sat[layer + 1], scalar(0))
                if mapmaximum(excessLj) > 0:
                    val = float(mapmaximum(excessLj))
                    self.theta[layer + 1] = ifthenelse(self.theta[layer + 1] > self.theta_sat[layer + 1],
                                                       self.theta_sat[layer + 1],
                                                       self.theta[layer + 1])
                    if float(val) > float(1e-02):
                        print("Corrected Percolation(), SAT exceeded, layer " + str(layer + 1) + ' by ' + str(val))

                # infil_z1 is not added here because already added to the layer below, See above: SW1
                percolation.append(getPercolation(self, layer, k_sat[layer], isPermeable=permeable))  # [mm]
                water_flux_z0 = infil_z1 + percolation[0]

                light_leached.append(getLeachedMass(self, layer, water_flux_z0, self.lightmass[layer],
                                                    sorption_model="linear", leach_model="mcgrath", gas=True,
                                                    debug=self.DEBUG, run=self.LCH))
                heavy_leached.append(getLeachedMass(self, layer, water_flux_z0, self.heavymass[layer],
                                                    sorption_model="linear", leach_model="mcgrath", gas=True,
                                                    debug=self.DEBUG, run=self.LCH))

                SW1b = self.theta[layer] * self.layer_depth[layer] - percolation[layer]
                self.theta[layer] = SW1b / self.layer_depth[layer]
                self.lightmass[layer] -= light_leached[layer]
                self.heavymass[layer] -= heavy_leached[layer]

                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err LCH, light")
                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err LCH, heavy")

                # RunOff Mass
                # Mass & delta run-off (RO)
                mass_runoff.append(getRunOffMass(self, precip, runoff_z0, self.lightmass[layer],
                                                 transfer_model="nu-mlm-ro", sorption_model="linear",
                                                 gas=True, run=self.ROM))
                mass_runoff.append(getRunOffMass(self, precip, runoff_z0, self.heavymass[layer],
                                                 transfer_model="nu-mlm-ro", sorption_model="linear",
                                                 gas=True, run=self.ROM))
                self.lightmass[layer] -= mass_runoff[0]  # light
                self.heavymass[layer] -= mass_runoff[1]  # heavy
                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err RO, light")

                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err RO, heavy")

                # Discharge due to runoff at the outlet
                runoff_m3 = runoff_z0 * cellarea() / 1000  # m3
                accu_runoff_m3 = accuflux(self.ldd_subs, runoff_m3)
                out_runoff_m3 = areatotal(accu_runoff_m3, self.outlet_multi)

            else:  # Layers 1, 2, 3 & 4

                if layer == (self.num_layers - 1):
                    permeable = self.bsmntIsPermeable

                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err Startz1, light")
                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err Startz1, heavy")

                excess = ifthenelse(self.theta[layer] > self.theta_sat[layer],
                                    self.theta[layer] - self.theta_sat[layer], scalar(0))
                if mapmaximum(excess) > scalar(0):
                    val = float(mapmaximum(excess))
                    if float(val) > float(1e-06):
                        print("Error at right before Percolation(), layer > 0, SAT exceeded at layer " + str(layer))
                    self.theta[layer] = ifthenelse(self.theta[layer] > self.theta_sat[layer], self.theta_sat[layer],
                                                   self.theta[layer])

                SW2 = self.theta[layer] * self.layer_depth[layer] + percolation[layer - 1]
                self.theta[layer] = SW2 / self.layer_depth[layer]
                # exceed = max(self.theta[layer] - self.theta_sat[layer], scalar(0))
                # self.report(exceed, 'outEXz' + str(layer))

                # recordInfiltration(self, percolation[layer - 2], layer)
                # self.report(SW, 'SWz' + str(layer))
                # if self.TEST_theta:
                #     checkMoisture(self, self.theta, 'athz')

                self.lightmass[layer] += light_leached[layer - 1]
                self.heavymass[layer] += heavy_leached[layer - 1]

                excess = ifthenelse(self.theta[layer] > self.theta_sat[layer],
                                    self.theta[layer] - self.theta_sat[layer], scalar(0))
                if mapmaximum(excess) > scalar(0):
                    val = float(mapmaximum(excess))
                    if val > float(1e-06):
                        print("Error at Percolation() layers: 1, 2, 3, "
                              "SAT exceeded, layer " + str(layer) + ' by ' + str(val))
                    self.theta[layer] = ifthenelse(self.theta[layer] > self.theta_sat[layer], self.theta_sat[layer],
                                                   self.theta[layer])

                percolation.append(getPercolation(self, layer, k_sat[layer], isPermeable=permeable))

                if layer < (len(self.layer_depth) - 1):  # layers: 1,2,3
                    sw_check_bottom = self.theta[layer + 1] * self.layer_depth[layer + 1] + percolation[layer]
                    exceed_mm = max(sw_check_bottom - self.theta_sat[layer + 1] * self.layer_depth[layer + 1],
                                    scalar(0))
                    if float(mapmaximum(exceed_mm)) > float(1e-03):
                        self.report(exceed_mm, 'outEXz' + str(layer + 1))

                light_leached.append(getLeachedMass(self, layer, percolation[layer], self.lightmass[layer],
                                                    sorption_model="linear", leach_model="mcgrath", gas=True,
                                                    debug=self.DEBUG, run=self.LCH))
                heavy_leached.append(getLeachedMass(self, layer, percolation[layer], self.heavymass[layer],
                                                    sorption_model="linear", leach_model="mcgrath", gas=True,
                                                    debug=self.DEBUG, run=self.LCH))

                SW3 = self.theta[layer] * self.layer_depth[layer] - percolation[layer]
                self.theta[layer] = SW3 / self.layer_depth[layer]

                self.lightmass[layer] -= light_leached[layer]
                self.heavymass[layer] -= heavy_leached[layer]
                if mapminimum(self.lightmass[layer]) < 0:
                    print("Err LCHz1, light")
                if mapminimum(self.heavymass[layer]) < 0:
                    print("Err LCHz1, heavy")

            if mapminimum(self.theta[layer]) < 0:
                print('Error at Percolation, Layer ' + str(layer))

            if self.TEST_LCH:
                recordLCH(self, light_leached[layer], layer)

            if self.TEST_IR:
                if layer == 0:
                    recordInfiltration(self, infil_z0, layer)
                    recordRunOff(self, runoff_z0)
                else:
                    recordInfiltration(self, percolation[layer - 1], layer)

            if self.TEST_PERC:
                recordPercolation(self, percolation[layer], layer)

        water_flux_z0_m3 = water_flux_z0 * cellarea() / 1000  # m3
        water_flux_z1_m3 = percolation[1] * cellarea() / 1000  # m3
        water_flux_z0_m3 = areatotal(water_flux_z0_m3, self.is_catchment)
        water_flux_z1_m3 = areatotal(water_flux_z1_m3, self.is_catchment)
        self.resW_accDPz0_m3_tss.sample(water_flux_z0_m3)
        self.resW_accDPz1_m3_tss.sample(water_flux_z1_m3)

        # Artificial drainage (relevant layer)
        drained_layers = [n for n, x in enumerate(self.drainage_layers) if x is True]  # <- list of indexes
        adr_layer = int(drained_layers[0])  # <- 13.05.2018, implements only one layer (i.e. z2)!
        assert adr_layer == 2
        cell_drainge_outflow = getArtificialDrainage(self, adr_layer)  # mm
        light_drained = getDrainMassFlux(self, adr_layer, self.lightmass[adr_layer], run=self.ADRM)  #
        heavy_drained = getDrainMassFlux(self, adr_layer, self.heavymass[adr_layer], run=self.ADRM)  #
        self.lightmass[adr_layer] -= light_drained
        self.heavymass[adr_layer] -= heavy_drained

        SW4 = self.theta[adr_layer] * self.layer_depth[adr_layer] - cell_drainge_outflow
        self.theta[adr_layer] = SW4 / self.layer_depth[adr_layer]

        if mapminimum(self.theta[adr_layer]) < 0:
            print('Error at ADR, Layer ' + str(adr_layer))

        if mapminimum(self.lightmass[adr_layer]) < 0:
            print("Err DR, light")
            self.report(self.lightmass[adr_layer], 'ADR_Mz')

        if mapminimum(self.heavymass[adr_layer]) < 0:
            print("Err DR, heavy")

        # Artificial drainage (Outlet discharge)
        cell_drain_z2_m3 = cell_drainge_outflow * cellarea() / 1000  # m3
        accu_drain_m3 = accuflux(self.ldd_subs, cell_drain_z2_m3)  # m3
        out_drain_m3 = areatotal(accu_drain_m3, self.outlet_multi)

        # Evapotranspiration
        etp = []
        evap = []
        transp = []
        etp_m3 = deepcopy(self.zero_map)
        evap_m3 = deepcopy(self.zero_map)
        transp_m3 = deepcopy(self.zero_map)
        depth_evap = self.layer_depth[0] + self.layer_depth[1]
        for layer in range(self.num_layers):
            act_evaporation_layer = deepcopy(self.zero_map)
            if layer < 2:
                pot_evapor_layer = pot_evapor * self.layer_depth[layer] / depth_evap
                act_evaporation_layer = getActualEvap(self, layer, pot_evapor_layer, run=self.ETP)
            # Evaporation
            SW5 = self.theta[layer] * self.layer_depth[layer] - act_evaporation_layer
            self.theta[layer] = SW5 / self.layer_depth[layer]
            evap.append(act_evaporation_layer)
            evap_m3 += evap[layer] * cellarea() / 1000  # m3

            # Transpiration
            act_transpir_layer = getActualTransp(self, layer, root_depth_tot, root_depth[layer],
                                                 pot_transpir, depletable_water, run=self.ETP)
            act_transpir_layer *= self.f_transp
            SW6 = self.theta[layer] * self.layer_depth[layer] - act_transpir_layer
            self.theta[layer] = SW6 / self.layer_depth[layer]
            transp.append(act_transpir_layer)
            transp_m3 += transp[layer] * cellarea() / 1000  # m3

            etp.append(act_transpir_layer + act_evaporation_layer)
            etp_m3 += etp[layer] * cellarea() / 1000  # m3

            if mapminimum(self.theta[layer]) < 0:
                print('Error at ETP, Layer ' + str(layer))

        evap_m3 = areatotal(evap_m3, self.is_catchment)
        transp_m3 = areatotal(transp_m3, self.is_catchment)
        self.resW_accEvap_m3_tss.sample(evap_m3)
        self.resW_accTransp_m3_tss.sample(transp_m3)

        # Lateral flow
        latflow_net = []  # every cell
        latflow_net_m3 = []
        catch_n_latflow_m3 = deepcopy(self.zero_map)  # Net
        cell_lat_outflow_m3 = deepcopy(self.zero_map)  # Only out

        latflow_outlet_mm = []
        latflow_cell_mm = []
        ligth_latflow = []
        heavy_latflow = []
        for layer in range(self.num_layers):
            # if layer < (self.num_layers - 1):

            # Get lateral flow upstream cells
            latflow_dict = getLateralFlow(self, layer, run=self.LF)
            latflow_cell_mm.append(latflow_dict['cell_outflow'])  # flux map
            self.theta[layer] = latflow_dict['new_moisture']  # state map

            # Outlet-cell flux Part II: add flux map to Part I.
            outlet_flux2_mm = ifthenelse(self.is_outlet, latflow_dict['cell_outflow'], deepcopy(self.zero_map))
            # latflow_outlet_mm.append(outlet_flux1_mm + outlet_flux2_mm)
            latflow_outlet_mm.append(outlet_flux2_mm)

            if mapmaximum(self.theta[layer]) > mapmaximum(self.theta_sat[layer]):
                val = mapmaximum(self.theta[layer]) - mapmaximum(self.theta_sat[layer])
                self.theta[layer] = ifthenelse(self.theta[layer] > self.theta_sat[layer], self.theta_sat[layer],
                                               self.theta[layer])
                if float(val) > float(1e-06):
                    print("getLateralFlow(), Corrected SAT excess, layer " + str(layer) + ' by ' + str(val))

            # else:  # Basement layer
            #     latflow_outlet_mm.append(deepcopy(self.zero_map))
            #     latflow_cell_mm.append(deepcopy(self.zero_map))

            ligth_latflow_dict = getLatMassFlux(self, layer, self.lightmass[layer], latflow_cell_mm[layer],
                                                debug=self.TEST_LFM, run=self.LFM)
            heavy_latflow_dict = getLatMassFlux(self, layer, self.heavymass[layer], latflow_cell_mm[layer],
                                                debug=self.TEST_LFM, run=self.LFM)
            ligth_latflow.append(ligth_latflow_dict['mass_loss'])
            heavy_latflow.append(heavy_latflow_dict['mass_loss'])

            self.lightmass[layer] = ligth_latflow_dict['new_mass']
            self.heavymass[layer] = heavy_latflow_dict['new_mass']
            cell_lat_outflow_m3 += latflow_outlet_mm[layer] * cellarea() / 1000  # m3, only out!

            if mapminimum(self.theta[layer]) < 0:
                print('Error at LF, Layer ' + str(layer))
            if mapminimum(self.lightmass[layer]) < 0:
                print("Err LF, light")
                self.report(self.lightmass[layer], 'LFM_Mz' + str(layer))
            if mapminimum(self.heavymass[layer]) < 0:
                print("Err LF, heavy")

        # Lateral flow (Outlet discharge)
        outlet_latflow_m3 = areatotal(cell_lat_outflow_m3, self.outlet_multi)  # Only outlet cells

        # Baseflow
        # Considering part of basement layer as completely saturated
        # baseflow_mm = (self.theta_sat[-2] * self.layer_depth[-2] * self.gw_factor) / self.k_g  # [mm/d]
        SWbsmt = self.theta[-1] * self.layer_depth[-1]
        baseflow_mm = SWbsmt / self.k_g  # [mm/d]
        SWbsmt -= baseflow_mm
        if mapminimum(SWbsmt) < 0:
            val = float(mapminimum(SWbsmt))
            if val < float(-1e-06):
                print("Negative Basement Soil Water by: " + str(val))
        self.theta[-1] = max(SWbsmt / self.layer_depth[-1], scalar(0))

        accu_baseflow_m3 = accuflux(self.ldd_subs, baseflow_mm * cellarea() / 1000)
        out_baseflow_m3 = areatotal(accu_baseflow_m3, self.outlet_multi)

        light_deg = []
        heavy_deg = []
        light_aged_deg = []
        heavy_aged_deg = []
        ch_storage_light = []
        ch_storage_heavy = []
        ch_storage_light_aged = []
        ch_storage_heavy_aged = []
        for layer in range(self.num_layers):
            # Temperature
            temp_dict = getLayerTemp(self, layer, bio_cover, temp_bare_soil)
            self.temp_surf_fin = temp_dict["temp_surface"]
            self.temp_fin[layer] = temp_dict["temp_layer"]

            # Degradation
            deg_light_dict = getMassDegradation(self, layer, self.lightmass[layer], self.light_aged[layer],
                                                frac="L", sor_deg_factor=1, fixed_dt50=self.fixed_dt50,
                                                deg_method='macro',
                                                bioavail=self.bioavail,
                                                debug=self.TEST_DEG, run=self.DEG)
            deg_heavy_dict = getMassDegradation(self, layer, self.heavymass[layer], self.heavy_aged[layer],
                                                frac="H", sor_deg_factor=1, fixed_dt50=self.fixed_dt50,
                                                deg_method='macro',
                                                bioavail=self.bioavail,
                                                debug=self.TEST_DEG, run=self.DEG)
            # self.report(deg_light_dict["mass_tot_new"], 'LoutZ' + str(layer))
            # self.report(deg_heavy_dict["mass_tot_new"], 'HoutZ' + str(layer))

            self.lightmass[layer] = deg_light_dict["mass_tot_new"]
            self.light_aged[layer] = deg_light_dict["mass_aged_new"]

            light_deg.append(deg_light_dict.get("mass_deg_aq") +
                             deg_light_dict.get("mass_deg_ads"))
            heavy_deg.append(deg_heavy_dict.get("mass_deg_aq") +
                             deg_heavy_dict.get("mass_deg_ads"))

            light_aged_deg.append(deg_light_dict["mass_deg_aged"])
            heavy_aged_deg.append(deg_heavy_dict["mass_deg_aged"])

            self.heavymass[layer] = deg_heavy_dict["mass_tot_new"]
            self.heavy_aged[layer] = deg_heavy_dict["mass_aged_new"]

            # Combined aged and fresh
            self.light_real[layer] = self.lightmass[layer] + self.light_aged[layer]
            self.heavy_real[layer] = self.heavymass[layer] + self.heavy_aged[layer]

            if mapminimum(self.lightmass[layer]) < 0:
                print("Err DEG, light")
                if layer == 0:
                    self.report(self.lightmass[layer], 'DEG_Mz' + str(layer))

            if mapminimum(self.heavymass[0]) < 0:
                print("Err DEG, heavy")

            # Change in mass storage after degradation - Pesticide Mass
            ch_storage_light.append(self.lightmass[layer] -
                                    self.lightmass_ini[layer])

            ch_storage_light_aged.append(self.light_aged[layer] -
                                         self.lightaged_ini[layer])

            ch_storage_heavy.append((self.heavymass[layer] -
                                    self.heavymass_ini[layer]))

            ch_storage_heavy_aged.append(self.heavy_aged[layer] -
                                         self.heavyaged_ini[layer])

            self.lightmass_ini[layer] = deepcopy(self.lightmass[layer])
            self.lightaged_ini[layer] = deepcopy(self.light_aged[layer])

            self.heavymass_ini[layer] = deepcopy(self.heavymass[layer])
            self.heavyaged_ini[layer] = deepcopy(self.heavy_aged[layer])

            self.delta[layer] = ((self.heavymass[layer] / self.lightmass[layer] - self.r_standard) /
                                 self.r_standard) * 1000  # [permille]

            self.delta_real[layer] = ((self.heavy_real[layer] / self.light_real[layer] - self.r_standard) /
                                      self.r_standard) * 1000  # [permille]

            self.delta_aged[layer] = ((self.heavy_aged[layer] / self.light_aged[layer] - self.r_standard) /
                                      self.r_standard) * 1000  # [permille]

        """ Layer analysis """
        if self.TEST_theta:
            for layer in range(self.num_layers):
                if layer == (self.num_layers - 1):
                    getLayerAnalysis(self, layer, percolation, latflow_outlet_mm, evap, transp,
                                     root_depth, out_baseflow_m3=out_baseflow_m3)
                else:
                    getLayerAnalysis(self, layer, percolation, latflow_outlet_mm, evap, transp,
                                     root_depth)

        # Update state variables
        # Change in storage - Moisture
        ch_storage = []
        ch_storage_m3 = deepcopy(self.zero_map)
        for layer in range(self.num_layers):
            ch_storage.append((self.theta[layer] * self.layer_depth[layer] * cellarea() / 1000) -
                              (self.theta_ini[layer] * self.layer_depth[layer] * cellarea() / 1000))
            self.theta_ini[layer] = deepcopy(self.theta[layer])
            ch_storage_m3 += ch_storage[layer]  # Reservoir storage (m3)

        if self.TEST_theta:
            getCatchmentStorage(self)
            getAverageMoisture(self)

        # Get Transect concentrations
        # Observed conc. is betw 2 and 8 ug/g dry soil (on transect)
        # Observed conc. can reach 20 ug/g dry soil (on single plot on application)

        # Record soil concentrations and isotopes
        if self.PEST:
            # Bio-available fraction (only)
            cell_mass = self.lightmass[0] + self.heavymass[0]
            cell_massXdelta = cell_mass * self.delta[0]
            reportSoilTSS(self, cell_mass, cell_massXdelta, 'north', type='bioavail')
            reportSoilTSS(self, cell_mass, cell_massXdelta, 'valley', type='bioavail')
            reportSoilTSS(self, cell_mass, cell_massXdelta, 'south', type='bioavail')

            # Aged fraction only
            cell_mass_aged = self.light_aged[0] + self.heavy_aged[0]
            cell_massXdelta_aged = cell_mass_aged * self.delta_aged[0]
            reportSoilTSS(self, cell_mass_aged, cell_massXdelta_aged, 'north', type='aged')
            reportSoilTSS(self, cell_mass_aged, cell_massXdelta_aged, 'valley', type='aged')
            reportSoilTSS(self, cell_mass_aged, cell_massXdelta_aged, 'south', type='aged')

            # Bio-available and aged fractions
            cell_mass_real = self.light_real[0] + self.heavy_real[0]
            cell_massXdelta_real = cell_mass_real * self.delta_real[0]
            soils_north = reportSoilTSS(self, cell_mass_real, cell_massXdelta_real, 'north', type='real')
            soils_valley = reportSoilTSS(self, cell_mass_real, cell_massXdelta_real, 'valley', type='real')
            soils_south = reportSoilTSS(self, cell_mass_real, cell_massXdelta_real, 'south', type='real')

            # Real mass catchment z0 and z+
            catch_light_real_z0 = areatotal(self.light_real[0], self.is_catchment)
            catch_heavy_real_z0 = areatotal(self.heavy_real[0], self.is_catchment)
            nor_real_z0 = areatotal(self.light_real[0] + self.heavy_real[0], self.is_north)
            val_real_z0 = areatotal(self.light_real[0] + self.heavy_real[0], self.is_valley)
            sou_real_z0 = areatotal(self.light_real[0] + self.heavy_real[0], self.is_south)

            catch_light_real_zX = deepcopy(self.zero_map)
            catch_heavy_real_zX = deepcopy(self.zero_map)

            # Aged mass catchment z0 and z+
            catch_light_aged_z0 = areatotal(self.light_aged[0], self.is_catchment)
            catch_heavy_aged_z0 = areatotal(self.heavy_aged[0], self.is_catchment)
            catch_light_aged_zX = deepcopy(self.zero_map)
            catch_heavy_aged_zX = deepcopy(self.zero_map)

            for layer in range(1, self.num_layers):
                catch_light_real_zX += areatotal(self.light_real[layer], self.is_catchment)
                catch_heavy_real_zX += areatotal(self.heavy_real[layer], self.is_catchment)
                catch_light_aged_zX += areatotal(self.light_aged[layer], self.is_catchment)
                catch_heavy_aged_zX += areatotal(self.heavy_aged[layer], self.is_catchment)

            # Real (z0, zX, heavy and light)
            reportSoilMass(self, "resM_light_real_z0", catch_light_real_z0)
            reportSoilMass(self, "resM_heavy_real_z0", catch_heavy_real_z0)
            reportSoilMass(self, "resM_light_real_zX", catch_light_real_zX)
            reportSoilMass(self, "resM_heavy_real_zX", catch_heavy_real_zX)

            reportSoilMass(self, "resM_lhr_nor_z0", nor_real_z0)
            reportSoilMass(self, "resM_lhr_val_z0", val_real_z0)
            reportSoilMass(self, "resM_lhr_sou_z0", sou_real_z0)

            # Aged (z0, zX, heavy and light)
            reportSoilMass(self, "resM_light_aged_z0", catch_light_aged_z0)
            reportSoilMass(self, "resM_heavy_aged_z0", catch_heavy_aged_z0)
            reportSoilMass(self, "resM_light_aged_zX", catch_light_aged_zX)
            reportSoilMass(self, "resM_heavy_aged_zX", catch_heavy_aged_zX)


        #####################
        # End of Model Loop #
        self.jd_cum += self.jd_dt  # updating JDcum, currently dt = 2 day

        ###################
        # Water Balance  ##
        ###################
        q_obs = timeinputscalar('q_obs_m3day.tss', nominal("outlet_v3"))
        # conc_outlet_obs = timeinputscalar('Conc_ugL.tss', nominal("outlet_v3"))
        # iso_outlet_obs = timeinputscalar('Delta_out.tss', nominal("outlet_v3"))

        # Total discharge
        tot_vol_disch_m3 = getTotalDischarge(out_runoff_m3,
                                             outlet_latflow_m3,
                                             out_drain_m3, baseflow=out_baseflow_m3)
        self.resW_accQ_m3_tss.sample(tot_vol_disch_m3)

        # Percolation Z0 and z1

        # percol_basement_m3 = percolation[-1] * cellarea() / 1000  # m3
        # out_percol_m3 = accuflux(self.ldd_subs, percol_basement_m3)
        # out_percol_m3 = areatotal(out_percol_m3, self.outlet_multi)

        # Evapotranspiration
        out_etp_m3 = accuflux(self.ldd_subs, etp_m3)
        out_etp_m3 = areatotal(out_etp_m3, self.outlet_multi)

        # Change in storage
        accu_ch_storage_m3 = accuflux(self.ldd_subs, ch_storage_m3)
        accu_ch_storage_m3 = areatotal(accu_ch_storage_m3, self.outlet_multi)

        # Cumulative
        # reportCumHydro(self, q_obs, out_runoff_m3, out_drain_m3, tot_rain_m3,
        #               out_etp_m3, outlet_latflow_m3, out_percol_m3=None)

        reportGlobalWaterBalance(self, tot_rain_m3, out_runoff_m3, out_drain_m3,
                                 outlet_latflow_m3, out_etp_m3, accu_ch_storage_m3,
                                 out_percol_m3=None,
                                 out_baseflow_m3=out_baseflow_m3)

        if self.TEST_theta and self.currentTimeStep() % 2 == 0:
            checkMoisture(self, self.theta, 'athz')

        ######################
        # Pesticide Balance ##
        ######################
        # Applied mass on catchment
        catch_app = areatotal(light_applied + heavy_applied, self.is_catchment)  #
        self.resM_accAPP_g_tss.sample(catch_app)
        reportTransectSinkTSS(self, 'APP_mass', self.cum_appZ0_g, 'north')
        reportTransectSinkTSS(self, 'APP_mass', self.cum_appZ0_g, 'valley')
        reportTransectSinkTSS(self, 'APP_mass', self.cum_appZ0_g, 'south')

        # Degradation
        light_deg_tot = deepcopy(self.zero_map)
        heavy_deg_tot = deepcopy(self.zero_map)
        for layer in range(1, self.num_layers):
            light_deg_tot += light_deg[layer]
            heavy_deg_tot += heavy_deg[layer]

        z0_deg_catch = areatotal(light_deg[0] + heavy_deg[0], self.is_catchment)
        z0_deg_nor = areatotal(light_deg[0] + heavy_deg[0], self.is_north)
        z0_deg_val = areatotal(light_deg[0] + heavy_deg[0], self.is_valley)
        z0_deg_sou = areatotal(light_deg[0] + heavy_deg[0], self.is_south)

        zX_deg_catch = areatotal(light_deg_tot + heavy_deg_tot, self.is_catchment)

        self.resM_accDEGzX_tss.sample(zX_deg_catch)
        self.resM_accDEGz0_tss.sample(z0_deg_catch)
        self.resM_accDEGz0nor_tss.sample(z0_deg_nor)
        self.resM_accDEGz0val_tss.sample(z0_deg_val)
        self.resM_accDEGz0sou_tss.sample(z0_deg_sou)
        reportTransectSinkTSS(self, 'DEG_mass', light_deg[0] + heavy_deg[0], 'north')
        reportTransectSinkTSS(self, 'DEG_mass', light_deg[0] + heavy_deg[0], 'valley')
        reportTransectSinkTSS(self, 'DEG_mass', light_deg[0] + heavy_deg[0], 'south')

        # self.cum_degZ0_g += z0_deg_catch
        # self.cum_degZ0_g_tss.sample(self.cum_degZ0_g)

        # Aged
        light_aged_tot = deepcopy(self.zero_map)
        heavy_aged_tot = deepcopy(self.zero_map)
        light_aged_deg_tot = deepcopy(self.zero_map)
        heavy_aged_deg_tot = deepcopy(self.zero_map)
        for layer in range(1, self.num_layers):
            light_aged_tot += self.light_aged[layer]
            heavy_aged_tot += self.heavy_aged[layer]
            light_aged_deg_tot += light_aged_deg[layer]
            heavy_aged_deg_tot += heavy_aged_deg[layer]

        z0_aged_catch = areatotal(self.light_aged[0] + self.heavy_aged[0], self.is_catchment)
        zX_aged_catch = areatotal(light_aged_tot + heavy_aged_tot, self.is_catchment)
        z0_aged_deg_catch = areatotal(light_aged_deg[0] + heavy_aged_deg[0], self.is_catchment)
        zX_aged_deg_catch = areatotal(light_aged_deg_tot + heavy_aged_deg_tot, self.is_catchment)

        self.resM_accAGEDz0_tss.sample(z0_aged_catch)
        self.resM_accAGEDzX_tss.sample(zX_aged_catch)
        self.resM_accAGED_DEGz0_tss.sample(z0_aged_deg_catch)
        self.resM_accAGED_DEGzX_tss.sample(zX_aged_deg_catch)
        reportTransectSinkTSS(self, 'AGE_mass', self.light_aged[0] + self.heavy_aged[0], 'north')
        reportTransectSinkTSS(self, 'AGE_mass', self.light_aged[0] + self.heavy_aged[0], 'valley')
        reportTransectSinkTSS(self, 'AGE_mass', self.light_aged[0] + self.heavy_aged[0], 'south')
        # self.cum_aged_deg_L_g += catch_aged_deg_light
        # self.cum_aged_deg_L_g_tss.sample(self.cum_aged_deg_L_g)

        # Volatilized
        catch_volat = areatotal(light_volat + heavy_volat, self.is_catchment)
        z0_volat_nor = areatotal(light_volat + heavy_volat, self.is_north)
        z0_volat_val = areatotal(light_volat + heavy_volat, self.is_valley)
        z0_volat_sou = areatotal(light_volat + heavy_volat, self.is_south)

        self.resM_accVOLATz0_tss.sample(catch_volat)
        self.resM_accVOLATz0nor_tss.sample(z0_volat_nor)
        self.resM_accVOLATz0val_tss.sample(z0_volat_val)
        self.resM_accVOLATz0sou_tss.sample(z0_volat_sou)
        reportTransectSinkTSS(self, 'VOLA_mass', light_volat + heavy_volat, 'north')
        reportTransectSinkTSS(self, 'VOLA_mass', light_volat + heavy_volat, 'valley')
        reportTransectSinkTSS(self, 'VOLA_mass', light_volat + heavy_volat, 'south')

        # Mass loss to run-off
        # Index: 0 <- light, Index: 2 <- heavy
        catch_runoff_light = areatotal(mass_runoff[0], self.is_catchment)
        catch_runoff_heavy = areatotal(mass_runoff[1], self.is_catchment)
        catch_runoff_mass = catch_runoff_light + catch_runoff_heavy
        nor_runoff = areatotal(mass_runoff[0] + mass_runoff[1], self.is_north)
        val_runoff = areatotal(mass_runoff[0] + mass_runoff[1], self.is_valley)
        sou_runoff = areatotal(mass_runoff[0] + mass_runoff[1], self.is_south)

        self.resM_accROz0_tss.sample(catch_runoff_mass)
        self.resM_accROz0nor_tss.sample(nor_runoff)
        self.resM_accROz0val_tss.sample(val_runoff)
        self.resM_accROz0sou_tss.sample(sou_runoff)
        reportTransectSinkTSS(self, 'ROFF_mass', mass_runoff[0] + mass_runoff[1], 'north')
        reportTransectSinkTSS(self, 'ROFF_mass', mass_runoff[0] + mass_runoff[1], 'valley')
        reportTransectSinkTSS(self, 'ROFF_mass', mass_runoff[0] + mass_runoff[1], 'south')

        # z0-mass leached
        catch_leach_light_z0 = areatotal(light_leached[0], self.is_catchment)
        catch_leach_heavy_z0 = areatotal(heavy_leached[0], self.is_catchment)
        z0_catch_leach = catch_leach_light_z0 + catch_leach_heavy_z0

        nor_leach_z0 = areatotal(light_leached[0] + heavy_leached[0], self.is_north)
        val_leach_z0 = areatotal(light_leached[0] + heavy_leached[0], self.is_valley)
        sou_leach_z0 = areatotal(light_leached[0] + heavy_leached[0], self.is_south)

        self.resM_accLCHz0_tss.sample(z0_catch_leach)
        self.resM_accLCHz0nor_tss.sample(nor_leach_z0)
        self.resM_accLCHz0val_tss.sample(val_leach_z0)
        self.resM_accLCHz0sou_tss.sample(sou_leach_z0)

        reportTransectSinkTSS(self, 'LCH_mass', light_leached[0] + heavy_leached[0], 'north')
        reportTransectSinkTSS(self, 'LCH_mass', light_leached[0] + heavy_leached[0], 'valley')
        reportTransectSinkTSS(self, 'LCH_mass', light_leached[0] + heavy_leached[0], 'south')
        # self.cum_lchZ0_L_g += catch_leach_light_z0
        # self.resM_cumLCHz0_L_g_tss.sample(self.cum_lchZ0_L_g)

        # z1-mass leached
        catch_leach_light_z1 = areatotal(light_leached[1] + heavy_leached[1], self.is_catchment)
        self.resM_accLCHz1_tss.sample(catch_leach_light_z1)

        # Basement-mass leached = zero, if no basement percolation
        # catch_leach_light_Bsmt = areatotal(light_leached[-1], self.is_catchment)
        # self.resM_accDP_L_tss.sample(catch_leach_light_Bsmt)

        # Artificial drained mass (layer z2)
        catch_drain_light = areatotal(light_drained, self.is_catchment)
        catch_drain_heavy = areatotal(heavy_drained, self.is_catchment)
        # catch_drain_heavy = areatotal(heavy_drained, self.is_catchment)
        self.resM_accADR_tss.sample(catch_drain_light + catch_drain_heavy)
        # catch_drain_heavy = areatotal(z1_heavy_drain, self.is_catchment)

        # Lateral flux at outlet cells
        # outlet_cell_lightflux = (ligth_latflow_outlet[0] + ligth_latflow_outlet[2] +
        #                          ligth_latflow_outlet[1] + ligth_latflow_outlet[3])
        # outlet_cell_lightflux = areatotal(outlet_cell_lightflux, self.outlet_multi)  # Sum only outlet cells


        # outlet_cell_heavyflux = (heavy_latflow_outlet[0] + heavy_latflow_outlet[2] +
        #                          heavy_latflow_outlet[1] + heavy_latflow_outlet[3])
        # outlet_cell_heavyflux = areatotal(outlet_cell_heavyflux, self.outlet_multi)  # Sum only outlet cells

        # Lateral flux (Required in MB)
        latflux_light_catch = deepcopy(self.zero_map)
        latflux_heavy_catch = deepcopy(self.zero_map)

        for layer in range(0, self.num_layers):
            latflux_light_catch += ligth_latflow[layer]
            latflux_heavy_catch += heavy_latflow[layer]

        catch_latflux_light = areatotal(latflux_light_catch, self.outlet_multi)  # Needed for MB
        catch_latflux_heavy = areatotal(latflux_heavy_catch, self.outlet_multi)  # Needed for MB
        self.resM_accLF_tss.sample(catch_latflux_light + catch_latflux_heavy)  # Reports the outlet-only loss

        # For mass balance Z0 layer
        z0_latflux = areatotal(ligth_latflow[0] + heavy_latflow[0], self.outlet_multi)

        # Baseflow flux
        # out_baseflow_light = areatotal(baseflow_light, self.is_catchment)
        # self.resM_accBF_L_tss.sample(out_baseflow_light)

        # Change in mass storage
        ch_storage_light_catch = deepcopy(self.zero_map)
        for layer in range(self.num_layers):
            ch_storage_light_catch += ch_storage_light[layer]

        # catch_ch_storage_light = areatotal(ch_storage_light_catch, self.is_catchment)
        z0_ch_storage_light = areatotal(ch_storage_light[0], self.is_catchment)
        z0_ch_storage_heavy = areatotal(ch_storage_heavy[0], self.is_catchment)
        z0_ch_storage = z0_ch_storage_light + z0_ch_storage_heavy
        # self.resM_accCHS_L_tss.sample(catch_ch_storage_light)

        # ch_storage_light_aged_catch = deepcopy(self.zero_map)
        # ch_storage_heavy_aged_catch = deepcopy(self.zero_map)
        # for layer in range(self.num_layers):
        #     ch_storage_light_aged_catch += ch_storage_light_aged[layer]
        #     ch_storage_heavy_aged_catch += ch_storage_heavy_aged[layer]

        z0_ch_storage_aged = areatotal(ch_storage_light_aged[0] + ch_storage_heavy_aged[0], self.is_catchment)
        # catch_ch_storage_light_aged = areatotal(ch_storage_light_aged_catch, self.is_catchment)
        # self.resM_accCHS_AGED_L_tss.sample(catch_ch_storage_light_aged)

        ####################
        # Outlet Pesticide #
        ####################
        # Total mass export
        outlet_light_export = (catch_runoff_light + catch_drain_light + catch_latflux_light)
        outlet_heavy_export = (catch_runoff_heavy + catch_drain_heavy + catch_latflux_heavy)
        self.resM_EXP_light_g_tss.sample(outlet_light_export)  # grams
        self.resM_EXP_heavy_g_tss.sample(outlet_heavy_export)  # grams

        conc_ugL = (outlet_light_export + outlet_heavy_export) * 1e6 / (tot_vol_disch_m3 * 1e3)
        conc_ROFF_ug_L = (catch_runoff_light + catch_runoff_heavy) * 1e6 / (tot_vol_disch_m3 * 1e3)
        conc_LF_ug_L = (catch_latflux_light + catch_latflux_heavy) * 1e6 / (tot_vol_disch_m3 * 1e3)
        conc_ADR_ug_L = (catch_drain_light + catch_drain_heavy) * 1e6 / (tot_vol_disch_m3 * 1e3)

        self.resM_oCONC_ugL_tss.sample(conc_ugL)  # ug/L
        self.resM_oCONC_ROFF_ugL_tss.sample(conc_ROFF_ug_L)  # ug/L
        self.resM_oCONC_LF_ugL_tss.sample(conc_LF_ug_L)  # ug/L
        self.resM_oCONC_ADR_ugL_tss.sample(conc_ADR_ug_L)  # ug/L

        # repCumOutMass(self,
        #               conc_outlet_obs, outlet_light_export,
        #               catch_latflux_light, catch_drain_light, catch_runoff_light,
        #               catch_volat_light, catch_deg_light)

        # Isotope signature - outlet
        out_delta = ((outlet_heavy_export / outlet_light_export - self.r_standard) /
                     self.r_standard) * 1000  # [permille]

        self.resM_outISO_d13C_tss.sample(out_delta)

        roff_delta = ((catch_runoff_heavy / catch_runoff_light - self.r_standard) /
                      self.r_standard) * 1000  # [permille]
        latflux_delta = ((catch_latflux_heavy / catch_latflux_light - self.r_standard) /
                         self.r_standard) * 1000  # [permille]
        drain_delta = ((catch_drain_heavy / catch_drain_light - self.r_standard) /
                       self.r_standard) * 1000  # [permille]

        self.resM_outISO_ROFF_d13C_tss.sample(roff_delta)
        self.resM_outISO_LF_d13C_tss.sample(latflux_delta)
        self.resM_outISO_ADR_d13C_tss.sample(drain_delta)

        reportGlobalPestBalance(self,
                                catch_app,
                                z0_deg_catch,
                                z0_aged_deg_catch,
                                catch_volat,
                                catch_runoff_mass,
                                z0_catch_leach,
                                # catch_drain_light,
                                z0_latflux,
                                z0_ch_storage,
                                z0_ch_storage_aged)

        # Total days with data (needed for mean calculations)
        self.days_cum += ifthenelse(q_obs >= 0, scalar(1), scalar(0))

        """
        Nash computations
        """
        # Analysis (NASH Discharge)
        # reportNashHydro(self, q_obs, tot_vol_disch_m3)

        # Analysis (NASH Pest)
        # if self.PEST:
        #     repNashOutConc(self, conc_outlet_obs, conc_ugL)
        #     repNashOutIso(self, iso_outlet_obs, out_delta,
        #                   roff_delta, latflux_delta, drain_delta)
        #
        #     repNashConcComposites(self,
        #                           soils_north['ave_conc'],
        #                           soils_valley['ave_conc'],
        #                           soils_south['ave_conc'])
        #     reptNashIsoComposites(self,
        #                           soils_north['d13C'],
        #                           soils_valley['d13C'],
        #                           soils_south['d13C'])


    def postmcloop(self):
        pass
        # names = ["q"]  # Discharge, Nash_Discharge
        # mcaveragevariance(names, self.sampleNumbers(), self.timeSteps())
        # aguila --timesteps=[170,280,2] q-ave q-var outlet_v3.map
        # percentiles = [0.25, 0.5, 0.75]
        # mcpercentiles(names, percentiles, self.sampleNumbers(), self.timeSteps())
        # aguila --quantiles=[0.25,0.75,0.25] --timesteps=[170,280,2] q



# Define models to run
problem = get_problem()
names = problem['names']
test = False
if test:
    samples = 2
    param_values = get_vector_test()  # Return a vector, with same values as names
    upper = np.ones(len(param_values)).tolist()
    # param_values = np.loadtxt('lhs_vectors.txt')
else:
    samples = 25
    upper = problem['upper']
    param_values = latin.sample(problem, samples)
    saveLHSmatrix(param_values)

firstTimeStep = start_jday()  # 166 -> 14/03/2016
nTimeSteps = 290  # 360

myAlteck16 = BeachModel("clone_nom.map", names, param_values, upper, staticDT50=False, test=test)
dynamicModel = DynamicFramework(myAlteck16, lastTimeStep=nTimeSteps,
                                firstTimestep=firstTimeStep)  # an instance of the Dynamic Framework
mcModel = MonteCarloFramework(dynamicModel, samples)

t0 = datetime.now()
# dynamicModel.run()
mcModel.run()
t1 = datetime.now()

duration = t1 - t0
tot_min = duration.total_seconds() / 60
print("Total minutes: ", tot_min)
print("Minutes/monte carlo", tot_min / int(samples))
print("Minutes/Yr: ", (duration.total_seconds() / 60) / (nTimeSteps - firstTimeStep) * 365)
# Visualization
# aguila 2\at0dC000.177 2\at1dC000.177
# aguila --scenarios='{2,1}' --multi=1x4  --timesteps=[175,179,2] aLEACH aLEACHz aLF aLFz
# aguila --scenarios='{2}'  --timesteps=[100,280,2] az0dC az1dC az2dC
# aguila --scenarios='{2}'  --timesteps=[2,280,2] aHeight aRDtot aCrop aPotETP akcb akcb1 akcmax
#  aguila --scenarios='{2}'  --timesteps=[2,360,2] aHeight aRDtot aCrop akcb aPotTRA aPotEVA
#  aguila --scenarios='{2,1,3}' --multi=1x4 --timesteps=[1,300,1] athz0 athz1 athz2 athz3
#  aguila --scenarios='{2,1,3}' --multi=1x4  --timesteps=[2,300,2] athz0 athz1 athz2 athz3
#  aguila --scenarios='{2,1,3}' --multi=1x4 --timesteps=[2,300,2] aObj1 aObj2
#  aguila --scenarios='{2,1,3}' thFCz2 thSATz2
#  aguila --scenarios='{2}' --timesteps=[1,300,1] aROm3 athz0 athz1 athz2 athz3
#  aguila --scenarios='{2}' --timesteps=[1,300,1] aROm3 aZ1LCH
# aguila --scenarios='{1}' --timesteps=[1,300,1] DPz1, DPz2

# Time series
# aguila 2\res_nash_q_m3.tss 6\res_nash_q_m3.tss
# aguila 2\resW_accStorage_m3.tss
# aguila 2\resM_norCONC.tss 2\resM_valCONC.tss 2\resM_souCONC.tss

