import scipy as sp
import pandas as pd
import numpy as np
import pickle


def get_data(gen="Gen1", v=None):
    if v is None:
        filename = "results" + gen + '_copy.p'
    else:
        filename = "results" + gen + '_copy' + v + '.p'
    res = pickle.load(open(filename, "rb"))
    return pd.DataFrame.from_dict(res, orient='index')


# def getTime(sep=","):
#     path = "Data/Time.csv"
#     obs = pd.read_csv(path, sep=sep)
#     return obs


def concat(row, name1, name2, sep=""):
    return row[name1] + sep + row[name2]


def IsoKGE(row, cst):
    if row['KGE-d13C'] >= cst:
        val = "KGE-d13C >  " + str(cst)
    elif row['KGE-d13C'] < cst:
        val = "KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def dt50(row, gen, ave):
    if row['Gen'] == gen:
        val = ave
    return val


def WithIsoKGE(row, cst):
    if row['KGE-d13C'] >= cst:
        val = "WC"
    elif row['KGE-d13C'] < cst:
        val = "NC"
    else:
        val = "Missed"
    return val

def BlkIso(row, cst):
    if row['KGE-d13C-blk'] >= cst:
        val = "WC"
    elif row['KGE-d13C-blk'] < cst:
        val = "NC"
    else:
        val = "Missed"
    return val


def TraIso(row, cst):
    if row['KGE-d13C-tra'] >= cst:
        val = "WC"
    elif row['KGE-d13C-tra'] < cst:
        val = "NC"
    else:
        val = "Missed"
    return val


def DetIso(row, cst):
    if row['KGE-d13C-det'] >= cst:
        val = "WC"
    elif row['KGE-d13C-det'] < cst:
        val = "NC"
    else:
        val = "Missed"
    return val


def LevelIso(row, metric):
    if row[metric] >= 0.8:
        val = metric + ' > ' + str(0.8)
    elif row[metric] >= 0.7:
        val = metric + ' > ' + str(0.7)
    elif row[metric] >= 0.6:
        val = metric + ' > ' + str(0.6)
    elif row[metric] >= 0.5:
        val = metric + ' > ' + str(0.5)
    else:
        val = metric + ' < ' + str(0.5)
    return val


def ModelIsoKGE(row, cst):
    if row['KGE-d13C'] >= cst and row['Model'] == 'var':
        val = "Var, KGE-d13C > " + str(cst)
    elif row['KGE-d13C'] >= cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C > " + str(cst)
    elif row['KGE-d13C'] < cst and row['Model'] == 'var':
        val = "Var, KGE-d13C < " + str(cst)
    elif row['KGE-d13C'] < cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def IsoKGEsoil(row, cst):
    if row['KGE-d13C-tra'] >= cst:
        val = "KGE-d13C >  " + str(cst)
    elif row['KGE-d13C-tra'] < cst:
        val = "KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def ConcKGEsoil(row, cst):
    if row['KGE-CONC-tra'] >= cst:
        val = "KGE-SM-tra >  " + str(cst)
    elif row['KGE-CONC-tra'] < cst:
        val = "KGE-SM-tra < " + str(cst)
    else:
        val = "Missed"
    return val


def ModelIsoKGEsoil(row, cst):
    if row['KGE-d13C-tra'] >= cst and row['Model'] == 'var':
        val = "Var, KGE-d13C > " + str(cst)
    elif row['KGE-d13C-tra'] >= cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C > " + str(cst)
    elif row['KGE-d13C-tra'] < cst and row['Model'] == 'var':
        val = "Var, KGE-d13C < " + str(cst)
    elif row['KGE-d13C-tra'] < cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def IsoKGEout(row, cst):
    if row['KGE-d13C_out'] >= cst:
        val = "KGE-d13C > " + str(cst)
    elif row['KGE-d13C_out'] < cst:
        val = "KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def ConcKGEout(row, cst):
    if row['KGE-CONC_out'] >= cst:
        val = "KGE-SM-out > " + str(cst)
    elif row['KGE-CONC_out'] < cst:
        val = "KGE-SM-out < " + str(cst)
    else:
        val = "Missed"
    return val


def ModelIsoKGEout(row, cst):
    if row['KGE-d13C_out'] >= cst and row['Model'] == 'var':
        val = "Var, KGE-d13C > " + str(cst)
    elif row['KGE-d13C_out'] >= cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C > " + str(cst)
    elif row['KGE-d13C_out'] < cst and row['Model'] == 'var':
        val = "Var, KGE-d13C < " + str(cst)
    elif row['KGE-d13C_out'] < cst and row['Model'] == 'fix':
        val = "Fix, KGE-d13C < " + str(cst)
    else:
        val = "Missed"
    return val


def ModelType(row, model):
    return model


# Confidence Interval
def get_ci(row, n, confidence, top):
    if top:
        ci = row['mean'] + row['sem'] * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    else:
        ci = row['mean'] - row['sem'] * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return ci


def check_negative(row):
    if row['low'] < 0:
        return row['min']
    else:
        return row['low']


"""
For getting sets and statistics, ES&T Figures
"""


def concat(row, plot):
    return plot.capitalize() + '-' + str(int(row['Jdays']))


# Extracting time series
def get_sets_bulk(name_list, path, p_b):
    # Get sim conc, convert mass, ug/g -> ug
    comp = ['nor', 'val', 'sou']
    sets = []
    for i in range(len(name_list)):  # Set name
        transects = []
        for tran in comp:
            # Append masses and conc.
            filename = "resM_" + tran + "CONC_real.tss"
            conc_name = tran + 'CONC'
            sim = pd.read_table(path + name_list[i] + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', conc_name],
                                header=None)

            mass_name = tran + "Mass"
            sim[mass_name] = sim[conc_name] * p_b['pbAve'] * 4.0 * 10.0 * 10 ** 3
            transects.append(sim)

            # Append deltas
            filename = "resM_" + tran + "d13C_real.tss"
            delta_name = tran + 'd13C'
            sim = pd.read_table(path + name_list[i] + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', delta_name],
                                header=None)
            transects.append(sim)

        # Merge all transects
        blk = reduce(lambda x, y: pd.merge(x, y, on='Jdays'), transects)

        # Bulk concentration
        conc_name = 'Conc_blk' + name_list[i][3:]
        blk[conc_name] = (blk['norCONC'] * blk['norMass'] +
                          blk['valCONC'] * blk['valMass'] +
                          blk['souCONC'] * blk['souMass']
                          ) / (blk['norMass'] + blk['valMass'] + blk['souMass'])

        iso_name = 'd13C_blk' + name_list[i][3:]
        blk[iso_name] = (blk['nord13C'] * blk['norMass'] +
                         blk['vald13C'] * blk['valMass'] +
                         blk['soud13C'] * blk['souMass']
                         ) / (blk['norMass'] + blk['valMass'] + blk['souMass'])

        blk = blk[0:121]
        blk = blk[['Jdays', conc_name, iso_name]]
        sets.append(blk)
    df = reduce(lambda left, right: pd.merge(left, right, on='Jdays'), sets)
    return df


# Extracting DT50, moisture and temp (TSS)
def get_sets(name_list, path, filename, vname):
    sets = []
    for i in range(len(name_list)):
        # Define variable name
        series_name = vname + name_list[i][8:]  # Variable + set's name
        # Get sim TSS
        sim = pd.read_table(path + name_list[i] + filename,
                            skiprows=4, delim_whitespace=True,
                            names=['Jdays', series_name],
                            header=None
                            )
        sim = sim[['Jdays', series_name]]
        sim = sim[0:121]
        sets.append(sim)
    df = reduce(lambda left, right: pd.merge(left, right, on='Jdays'), sets)
    return df


# Computing statistics from extracted sets
def get_stats_bulk(df, measure, cst):
    n = len(np.array(df.iloc[0, 1:]))
    df['mean'] = df.iloc[:, 1:n + 1].mean(axis=1)
    df['min'] = df.iloc[:, 1:n + 1].min(axis=1)
    df['max'] = df.iloc[:, 1:n + 1].max(axis=1)
    df['sem'] = df.iloc[:, 1:n + 1].sem(axis=1)
    df['sd'] = df.iloc[:, 1:n + 1].std(axis=1)
    df['high'] = df['mean'] + 2. * df['sd']
    df['low'] = df['mean'] - 2. * df['sd']

    if measure == "Conc":
        df['low'] = df.apply(check_negative, axis=1)

    loc = "B"
    df['IDcal'] = df.apply(lambda row: concat(row, loc), axis=1)
    df['Type'] = df.apply(lambda row: ModelType(row, cst), axis=1)
    return df[['Jdays', 'mean', 'high', 'low', 'max', 'min', 'sd', 'IDcal', 'Type']]


def get_stats_df(df):
    n = len(np.array(df.iloc[0, 1:]))
    df['mean'] = df.iloc[:, 1:n + 1].mean(axis=1)
    df['min'] = df.iloc[:, 1:n + 1].min(axis=1)
    df['max'] = df.iloc[:, 1:n + 1].max(axis=1)
    df['sem'] = df.iloc[:, 1:n + 1].sem(axis=1)
    df['sd'] = df.iloc[:, 1:n + 1].std(axis=1)
    df['high'] = df['mean'] + 2. * df['sd']
    df['low'] = df['mean'] - 2. * df['sd']
    df['low'] = df.apply(check_negative, axis=1)
    return df[['Jdays', 'mean', 'sd', 'high', 'low', 'max', 'min']]


"""
Metrics
"""


def get_sets_metrics(name_list, path, filename, vname):
    sets = []
    for i in range(len(name_list)):
        # Define variable name
        series_name = vname + name_list[i][3:]  # Variable + set's name
        # Get sim TSS
        sim = pd.read_table(path + name_list[i] + filename,
                            skiprows=4, delim_whitespace=True,
                            names=['Jdays', 'sim'],
                            header=None
                            )
        if vname is not "PER":
            sim[series_name] = sim['sim'].cumsum()

            if vname is "VOL":
                sim[series_name] *= 100

                # TODO: Add heavy fraction for mass exports
                #             if vname == "EXP":
                #                 filename = "resM_EXP_heavy_g.tss"
                #                 h = pd.read_table(name_list[i] + filename,
                #                        skiprows=4, delim_whitespace=True,
                #                                     names=['Jdays', 'simh'],
                #                                     header=None)
                # Cumsum, Merge and add heavy

        else:
            # TODO: Add heavy fraction for persistent mass
            sim[series_name] = sim['sim']

        sim = sim[['Jdays', series_name]]
        sim = sim[0:121]
        sets.append(sim)
    df = reduce(lambda left, right: pd.merge(left, right, on='Jdays'), sets)
    return df


def roofed(row):
    if row['value'] > 100.:
        val = 100.
    else:
        val = row['value']
    return val


def pct_df(df, metric, app, cst='WC', high='high', low='low'):
    sim = pd.merge(app, df, how='inner', on='Jdays')

    prct = 100.
    sim[str(metric + '_95%')] = prct * (sim[high] / sim['Capp'])
    sim[str(metric + '_50%')] = prct * (sim['mean'] / sim['Capp'])
    sim[str(metric + '_05%')] = prct * (sim[low] / sim['Capp'])

    sim = sim.loc[5:, ]  # Get rid of divisions by zero

    sim = sim[['Jdays', str(metric + '_95%'), str(metric + '_50%'), str(metric + '_05%')]]
    sim = pd.melt(sim, id_vars=['Jdays'],
                  value_vars=[str(metric + '_95%'), str(metric + '_50%'), str(metric + '_05%')])
    sim['Type'] = sim.apply(lambda row: ModelType(row, cst), axis=1)

    sim['value'] = sim.apply(lambda row: roofed(row), axis=1)

    sim['Jdays'] = sim['Jdays'] - 171
    return sim


def count_conc(row):
    if row['high'] >= row['ug.g']:
        if row['low'] <= row['ug.g']:
            value = 1.
        else:
            value = 0.
    else:
        value = 0.
    return value


def count_delta(row):
    if row['high'] >= row['d13C']:
        if row['low'] <= row['d13C']:
            value = 1.
        else:
            value = 0.
    else:
        value = 0.
    return value
