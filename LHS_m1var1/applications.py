# -*- coding: utf-8 -*-
from pcraster.framework import *


def getApplications(model, fa_cr, plot_codes, massunit='g'):
    """

    :param model:
    :param fa_cr:
    :param plot_codes:
    :param massunit:
    :return: # default mass is grams (to each pixel)
    """
    if massunit == 'g':
        factor = 1  # grams
    else:
        factor = 10 ** 6  # micro grams

    # Applications Mass
    # Product concentration (active ing.)
    double = scalar(2.0)  # ~ Dosage for corn when growing beet
    d_gold = scalar(915) * factor  # g/L S-met # * 10 ** 6  # ug/L S-met
    m_gold = scalar(960) * factor  # g/L S-met # * 10 ** 6  # ug/L

    # Dosages # L/Ha * 1Ha/1000m2 = L/m2
    d_beet = None
    d_corn = scalar(2.1) * 1 / 10 ** 4  # 2.2 L/Ha * 2 Ha / 10000 m2
    m_beet = scalar(0.6) * 1 / 10 ** 4
    m_corn = scalar(2.0) * 1 / 10 ** 4
    #
    m_beet_Friess = scalar(0.6) * 1 / 10 ** 4 * (double)  # 0.6 L/Ha * 2 Ha / 10000 m2 = L/m2
    m_beet_Mathis = scalar(0.6) * 1 / 10 ** 4 * (double)
    m_beet_Burger = scalar(0.6) * 1 / 10 ** 4 * (double + 1)  #
    m_beet_Kopp = scalar(0.6) * 1 / 10 ** 4 * (double + 1)

    app_conc = (  # [mass-unit/m2]
        ifthenelse(fa_cr == 1111,  # 1111 (Friess, Beet)
                   m_beet_Friess * m_gold * model.mask,
                   ifthenelse(fa_cr == 1122,  # 1112 (Friess-Corn),
                              m_corn * m_gold * model.mask,
                              ifthenelse(fa_cr == 1212,  # 1212 (Speich-Corn),
                                         m_corn * m_gold * model.mask,
                                         ifthenelse(fa_cr == 1312,  # 1312 (Mahler-Corn),
                                                    m_corn * m_gold * model.mask,
                                                    ifthenelse(fa_cr == 1412,  # 1412 (Schmitt-Corn)
                                                               d_corn * d_gold * model.mask,
                                                               ifthenelse(fa_cr == 1511,  # 1511 (Burger-Beet)
                                                                          m_beet_Burger * m_gold * model.mask,
                                                                          # 1711 (Mathis-Beet),
                                                                          ifthenelse(fa_cr == 1711,
                                                                                     m_beet_Mathis * m_gold * model.mask,
                                                                                     # 1611 (Kopp-Beet)
                                                                                     ifthenelse(
                                                                                         fa_cr == 1611,
                                                                                         m_beet_Kopp * m_gold * model.mask,
                                                                                         0 * model.mask))))))))
    )

    # Pesticide applied (mass-unit) on Julian day 171
    # March 20th, Fries
    app0 = ifthenelse(fa_cr == 1111, 1 * app_conc * cellarea(),  # Friess early, Beet
                      ifthenelse(fa_cr == 1112, 1 * app_conc * cellarea(),  # Friess early, Corn
                                 0 * app_conc * cellarea()))

    # March 26th, Mathis
    app1 = ifthenelse(fa_cr == 1711, 1 * app_conc * cellarea(),  # 1711 (Mathis-Beet)
                      0 * app_conc * cellarea())

    # Pesticide applied (mass-unit) on Julian day 196 (April 13, 2016).
    # April 13, Kopp and Burger
    app2 = ifthenelse(fa_cr == 1511, 1 * app_conc * cellarea(),  # 1511 (Burger-Beet)
                      ifthenelse(fa_cr == 1611, 1 * app_conc * cellarea(),  # 1611 (Kopp-Beet),
                                 0 * app_conc * cellarea()))

    # 2nd App, Friess on Julian day 200 (April...).
    Friess2ndApp = m_beet_Friess * m_gold * model.mask
    app3 = ifthenelse(plot_codes == 16, 1 * Friess2ndApp * cellarea(),  # (2nd Friess-Beet, app Day 200, North)
                      ifthenelse(plot_codes == 21, 1 * Friess2ndApp * cellarea(),
                                 # (2nd Friess-Beet, app Day 200, Valley)
                                 0 * app_conc * cellarea()))

    # Pesticide applied (mass-unit) on Julian day 213 (April 30, 2016).
    app4 = ifthenelse(fa_cr == 1511, 1 * app_conc * cellarea(),  # 1511 (2nd Burger-Beet, app Day 213 (209-216)
                      0 * app_conc * cellarea())

    # Pesticide applied (mass-unit) on Julian day 238 (May 25, 2016).
    # May 25, Schmidt and Speich, and (out of transect): Friess and Mahler
    # Note: Speich could be 2 weeks later.
    app5 = ifthenelse(fa_cr == 1112, 1 * app_conc * cellarea(),  # 1112 (Friess-Corn)
                      ifthenelse(fa_cr == 1412, 1 * app_conc * cellarea(),
                                 # 1412 (Schmitt-Corn),
                                 ifthenelse(fa_cr == 1312, 1 * app_conc * cellarea(),
                                            # 1312 (Mahler-Corn)
                                            0 * app_conc * cellarea())))
    # Speich on Day 245
    app6 = ifthenelse(fa_cr == 1212, 1 * app_conc * cellarea(),  # 1212 (Speich-Corn)
                      0 * app_conc * cellarea())

    return [app0, app1, app2, app3, app4, app5, app6]  # default mass is grams
