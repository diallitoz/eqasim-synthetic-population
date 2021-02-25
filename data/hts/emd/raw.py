import simpledbf
from tqdm import tqdm
import pandas as pd
import os

"""
This stage loads the raw data of the MEL HTS (EMD).
"""

MENAGES_COLUMNS = [
    "IDM4", "ZFM", "ECH", "GM1", "STM", "M6", "M14", "M21", "M22", "COE0"
]

PERSONNES_COLUMNS = [
    "IDP4", "ZFP", "ECH", "PER", "GP1", "STP", "P2", "P3", "P4", "P7", "P8", "P9", "P10", "PCSC", "PCSD", "P12", "P14",
    "P15","DP15","P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23", "P24", "COE1"
]

DEPLACEMENTS_COLUMNS = [
    "IDD4", "ZFD", "ECH", "PER", "NDEP", "GD1", "D2A", "D2B", "D3", "GDO1", "D4", "D5A", "D5B", "D6", "D7",
    "GDD1", "D8", "D9", "D10", "D11", "D12", "MODP", "TYPD"
]

TRAJETS_COLUMNS = [
    "IDT4", "ZFT", "ECH", "PER", "NDEP", "T1", "GT1", "T2", "T3", "T4", "GTO1", "T5", "GTD1", "T6", "T7", "T8", "T8A",
    "T8B", "T9", "T10", "T11", "T12", "T13"
]

def configure(context):
    context.config("data_path")

def execute(context):
    df_menages = pd.read_csv(
        "%s/emd_2016/Quetelet/menages.csv" % context.config("data_path"),
        sep=",", encoding="latin1", usecols=MENAGES_COLUMNS,
        dtype={"ZFM": str, "ECH": str}
    )

    df_personnes = pd.read_csv(
        "%s/emd_2016/Quetelet/personnes.csv" % context.config("data_path"),
        sep=",", encoding="latin1", usecols=PERSONNES_COLUMNS,
        dtype={"ZFP": str, "ECH": str, "PER": str}
    )

    df_deplacements = pd.read_csv(
        "%s/emd_2016/Quetelet/deplacements.csv" % context.config("data_path"),
        sep=",", encoding="latin1", usecols=DEPLACEMENTS_COLUMNS,
        dtype={"ZFD": str, "ECH": str, "PER": str, "NDEP": str, "D4": str, "D8" : str}
    )

    df_trajets = pd.read_csv(
        "%s/emd_2016/Quetelet/trajets.csv" % context.config("data_path"),
        sep=",", encoding="latin1", usecols=TRAJETS_COLUMNS,
        dtype={"ZFT": str, "ECH": str, "PER": str, "NDEP": str, "T1": str}
    )

    return df_menages, df_personnes, df_deplacements, df_trajets

def validate(context):
    for name in ("menages.csv", "personnes.csv", "deplacements.csv", "trajets.csv"):
        if not os.path.exists("%s/emd_2016/Quetelet/%s" % (context.config("data_path"), name)):
            raise RuntimeError("File missing from EMD: %s" % name)

    return [
        os.path.getsize("%s/emd_2016/Quetelet/menages.csv" % context.config("data_path")),
        os.path.getsize("%s/emd_2016/Quetelet/personnes.csv" % context.config("data_path")),
        os.path.getsize("%s/emd_2016/Quetelet/deplacements.csv" % context.config("data_path")),
        os.path.getsize("%s/emd_2016/Quetelet/trajets.csv" % context.config("data_path"))
    ]