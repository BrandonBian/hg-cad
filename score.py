import pandas as pd
from sklearn.metrics import cohen_kappa_score

# read the elliot.csv and bodia.csv files into DataFrames
df_elliot = pd.read_csv("elliot.csv")
df_bodia = pd.read_csv("bodia.csv")

# select the columns of interest
df_elliot_selected = df_elliot[["Assembly_Name", "Body_Name", "Label_Tier2"]]
df_bodia_selected = df_bodia[["Assembly_Name", "Body_Name", "Label_Tier2"]]


# Merge the two dataframes on the label id
df_merged = pd.merge(df_elliot_selected[['Assembly_Name', 'Body_Name', 'Label_Tier2']],
                     df_bodia_selected[['Assembly_Name', 'Body_Name', 'Label_Tier2']],
                     on=['Assembly_Name', 'Body_Name'],
                     suffixes=('_elliot', '_bodia'))

# Rename the columns to prepare for computing kappa score
df_merged = df_merged.rename(columns={'Label_Tier2_elliot': 'elliot', 'Label_Tier2_bodia': 'bodia'})

# Calculate the kappa score
kappa_score = cohen_kappa_score(df_merged['elliot'], df_merged['bodia'])

print(kappa_score)