import numpy as np
import pandas as pd
import os

"""
This stages loads a file containing all spatial codes in France and how
they can be translated into each other. These are mainly IRIS, commune,
departement and région.
"""

YEAR = 2017
SOURCE = "codes_%d/reference_IRIS_geo%d.xls" % (YEAR, YEAR)

##95 communes of MEL
commune95_ids = ["59350", "59005", "59011", "59013", "59017", "59025", "59044", "59051", "59052", "59056", "59088", "59090",
               "59098", "59106", "59128", "59133", "59143", "59146", "59152", "59163", "59173", "59670", "59193", "59195",
               "59196", "59201", "59202", "59208", "59220", "59247", "59250", "59252", "59256", "59257", "59275", "59278",
               "59279", "59281", "59286", "59299", "59303", "59316", "59317", "59320", "59328", "59332", "59339", "59343",
               "59346", "59352", "59356", "59360", "59367", "59368", "59371", "59378", "59386", "59388", "59410", "59421",
               "59426", "59437", "59457", "59458", "59470", "59477", "59482", "59487", "59507", "59508", "59512", "59522",
               "59523", "59524", "59527", "59550", "59553", "59560", "59566", "59585", "59598", "59599", "59602", "59609",
               "59611", "59009", "59636", "59643", "59646", "59648", "59650", "59653", "59656", "59658", "59660"]

##85 communes of MEL at 31/12/2016
commune_ids = ["59350", "59013", "59017", "59044", "59051", "59056", "59090", "59098", "59106", "59128", "59143",
               "59146", "59152", "59163", "59173", "59670", "59193", "59195", "59196", "59201", "59202", "59208",
               "59220", "59247", "59250", "59252", "59256", "59275", "59278", "59279", "59281", "59286", "59299",
               "59303", "59316", "59317", "59320", "59328", "59332", "59339", "59343", "59346", "59352", "59356",
               "59360", "59367", "59368", "59378", "59386", "59388", "59410", "59421", "59426", "59437", "59457",
               "59458", "59470", "59482", "59507", "59508", "59512", "59522", "59523", "59524", "59527", "59550",
               "59553", "59560", "59566", "59585", "59598", "59599", "59602", "59609", "59611", "59009", "59636",
               "59643", "59646", "59648", "59650", "59653", "59656", "59658", "59660"]

def configure(context):
    context.config("data_path")

    context.config("regions", ["32"])
    context.config("departments", ["59"])
    context.config("communes", commune_ids)

def execute(context):
    # Load IRIS registry
    df_codes = pd.read_excel(
        "%s/%s" % (context.config("data_path"), SOURCE),
        skiprows = 5, sheet_name = "Emboitements_IRIS"
    )[["CODE_IRIS", "DEPCOM", "DEP", "REG"]].rename(columns = {
        "CODE_IRIS": "iris_id",
        "DEPCOM": "commune_id",
        "DEP": "departement_id",
        "REG": "region_id"
    })

    df_codes["iris_id"] = df_codes["iris_id"].astype("category")
    df_codes["commune_id"] = df_codes["commune_id"].astype("category")
    df_codes["departement_id"] = df_codes["departement_id"].astype("category")
    df_codes["region_id"] = df_codes["region_id"].astype(int)

    # Filter zones
    requested_regions = list(map(int, context.config("regions")))
    requested_departments = list(map(str, context.config("departments")))
    requested_communes = list(map(str, context.config("communes")))

    if len(requested_regions) > 0:
        df_codes = df_codes[df_codes["region_id"].isin(requested_regions)]

    if len(requested_departments) > 0:
        df_codes = df_codes[df_codes["departement_id"].isin(requested_departments)]

    if len(requested_communes) > 0:
        df_codes = df_codes[df_codes["commune_id"].isin(requested_communes)]

    df_codes["iris_id"] = df_codes["iris_id"].cat.remove_unused_categories()
    df_codes["commune_id"] = df_codes["commune_id"].cat.remove_unused_categories()
    df_codes["departement_id"] = df_codes["departement_id"].cat.remove_unused_categories()

    return df_codes

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), SOURCE)):
        raise RuntimeError("Spatial reference codes are not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), SOURCE))
