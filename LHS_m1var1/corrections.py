
from pcraster import *
from pcraster._pcraster import *
from pcraster.framework import *
import os

print(os.getcwd())

"""
Adjusting z2 depth
"""
dem = readmap("dem_slope")  # 192 - 231 m a.s.l
zero_map = dem - dem  # Zero map to generate scalar maps
datum_depth = (dem - mapminimum(dem)) * scalar(10 ** 3)  # mm

# z2_factor = 0.8
# z0 = zero_map + 10
# z1 = zero_map + 300  # mm
# z2 = (datum_depth + 610 - z0 - z1)*z2_factor  # mm (300mm at outlet) * z2_factor
# tot_depth = z0 + z1 + z2
#
# aguila(datum_depth, z2)

"""
Converting clone.map to nominal type 
"""
# To assign ordinal, use "order(expression)"
# oldclone = readmap("clone")
# dem = readmap("dem_slope")
# res = boolean(dem)
# res = nominal(res)
# report(res, 'clone_nom.map')  # stores a ".map" file

"""
Correcting the outlet location
"""
# res = readmap("zzTest") # the result of the accuflux function (via ldd)
# outlet = ifthenelse(res == mapmaximum(res), nominal(2), nominal(0))
# report(outlet, 'outlet_true.map')

"""
fields_cover.map

Replacing missing values in landuse (due to ArcGis Errors"
with landuse = 5 (check again later)
"""
# fields = readmap("landuse")
# aguila(fields)
# fields_true = cover(fields, 5)
# dem = readmap("dem_slope")
# res = boolean(dem)
# fields_fin = ifthen(res, fields_true)  # Reduce the map to the catchment extent
# report(fields_fin, 'fields_cover.map')  # store a ".map" file


"""
Creating Nominal Transect Maps
"""
# obs = readmap("weekly_ord")
# trans = ifthenelse(obs <= 30, nominal(2), ifthenelse(obs <= 55, nominal(2), ifthenelse(obs <= 81, nominal(3), nominal(4))))
# report(trans, 'weekly_smp')
# aguila("weekly_smp")