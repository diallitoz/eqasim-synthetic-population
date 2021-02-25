import geopandas as gpd
import pandas as pd
import data.hts.hts as hts
import data.income.municipality as muniInc
import numpy as np

"""
This stage filters out EMD observations which live or work outside of MEL.
"""

def configure(context):
    context.config("data_path")
    context.stage("data.hts.emd.cleaned")
    context.stage("data.spatial.codes")

def execute(context):
    df_codes = context.stage("data.spatial.codes")
    requested_communes = df_codes["commune_id"].unique()
    df_households, df_persons, df_trips = context.stage("data.hts.emd.cleaned")

    # Filter trips
    df_shapes = gpd.read_file("%s/emd_2016/Quetelet/shape_files/epsg_2154/mel_ZF_epsg_2154.shp" % context.config("data_path"))
    df_shapes["ZFIN2016F"] = df_shapes["ZFIN2016F"].astype(np.int).astype("category")
    requested_zones_fines = df_shapes["ZFIN2016F"].unique()

    print(requested_zones_fines)

    print("Exemple of person's ZF \n", df_persons["zone_fine_id"])

    # Filter for non-residents taking the
    # Zone fines as living area
    f = df_persons["commune_id"].isin(requested_communes)
    df_persons = df_persons[f]
    print("df_persons size after communes filtering ", len(df_persons))

    f_zf = df_persons["zone_fine_id"].isin(requested_zones_fines)
    df_persons = df_persons[f_zf]

    print("df_persons size after ZF filtering ", len(df_persons))

    # Filter for people going outside of the area (because they have NaN distances)
    remove_ids = set()

    remove_ids |= set(df_trips[
        ~df_trips["origin_commune_id"].isin(requested_communes) | ~df_trips["destination_commune_id"].isin(requested_communes)
    ]["person_id"].unique())

    remove_ids |= set(df_trips[
        ~df_trips["origin_zone_fine_id"].isin(requested_zones_fines) | ~df_trips["destination_zone_fine_id"].isin(requested_zones_fines)
        ]["person_id"].unique())

    remove_ids |= set(df_persons[
        ~df_persons["commune_id"].isin(requested_communes)
    ])

    df_persons = df_persons[~df_persons["person_id"].isin(remove_ids)]
    print("df_persons size after remove_ids filtering ", len(df_persons))

    # Only keep trips and households that still have a person
    df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
    df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

    # Finish up

    df_households = df_households[hts.HOUSEHOLD_COLUMNS + ["commune_id"]]
    df_persons = df_persons[hts.PERSON_COLUMNS + ["commune_id"] + ["zone_fine_id"]]
    df_trips = df_trips[hts.TRIP_COLUMNS + ["euclidean_distance"] + ["routed_distance"] + ["origin_commune_id"] + ["destination_commune_id"] + ["origin_zone_fine_id"] + ["destination_zone_fine_id"]]

    hts.check(df_households, df_persons, df_trips)

    print("df_persons size after checking stage filtering ", len(df_persons), " persons with ", len(df_trips), " trips")

    #export_csv_households = df_households.to_csv(r'./output/df_households_MEL.csv', index=None, header=True)
    #export_csv_persons = df_persons.to_csv(r'./output/df_persons_MEL.csv', index=None, header=True)
    #export_csv_trips = df_trips.to_csv(r'./output/df_trips_MEL.csv', index=None, header=True)

    return df_households, df_persons, df_trips
