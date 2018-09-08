import seaborn as sns
sns.set()

tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips)

f = g.fig