import simpledbf
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

"""
This stage loads the raw data from the French population census.
"""

def configure(context):
    context.stage("data.spatial.codes")

    context.config("data_path")
    context.config("census_path", "rp_2015/FD_INDCVIZB_2015.dbf")# Change "A" to "B" correspond to the study area

COLUMNS = [
    "CANTVILLE", "NUMMI", "AGED",
    "COUPLE", "CS1",
    "DEPT", "ETUD", "ILETUD",
    "ILT", "IPONDI", "IRIS",
    "REGION", "SEXE",
    "TACT", "TRANS",
    "VOIT", "DEROU"
]

def execute(context):
    df_codes = context.stage("data.spatial.codes")
    # requested_departements = df_codes["departement_id"].unique()
    requested_communes = df_codes["commune_id"].unique()

    # Search Communes from CantonVille for filter populations within MEL because census data are provided by CANTONVILLE
    df_population = pd.read_excel(
        "%s/rp_2015/table-appartenance-geo-communes-17.xls" % context.config("data_path"),
        skiprows=5, sheet_name="COM"
    )[["CODGEO", "DEP", "REG", "ARR", "CV"]]  ##CODGEO=COMMUNES and CV=CANTONVILLE

    df_population = df_population[df_population["CODGEO"].isin(requested_communes)]

    cv_ids = set(np.unique(df_population["CV"]))

    ##Canton ou ville MEL : {'5902', '5933', '5924', '5997', '5926', '5904', '5918', '5937', '5922', '5913', '5936', '5940', '5928', '5925', '5998', '5923', '5999'}

    print("CANTVILLE of the MEL: ...")
    print(cv_ids)

    table = simpledbf.Dbf5("%s/%s" % (context.config("data_path"), context.config("census_path")))
    records = []

    with context.progress(total = 4320619, label = "Reading census ...") as progress:
        for df_chunk in table.to_dataframe(chunksize = 10240):
            progress.update(len(df_chunk))

            #df_chunk = df_chunk[df_chunk["DEPT"].isin(requested_departements)]
            df_chunk = df_chunk[df_chunk["CANTVILLE"].isin(cv_ids)]
            df_chunk = df_chunk[COLUMNS]

            if len(df_chunk) > 0:
                records.append(df_chunk)

    pd.concat(records).to_hdf("%s/census.hdf" % context.path(), "census")

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("census_path"))):
        raise RuntimeError("RP 2015 data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("census_path")))
