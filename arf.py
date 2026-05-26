import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datetime import datetime, timedelta

# =====================================================
# LOAD DATA
# =====================================================

topo = pd.read_csv("newtopography.csv")
soil = pd.read_csv("newsoil.csv")
twi = pd.read_csv("newtwi.csv")
drain = pd.read_csv("newdrainage.csv")

ndbi = pd.read_csv("newndbi_yearly.csv")
ibi = pd.read_csv("newibi_yearly.csv")

ndvi_2019 = pd.read_csv("newndvi_2019.csv")
ndvi_2024 = pd.read_csv("newndvi_2024.csv")

mndwi_2019 = pd.read_csv("newmndwi_2019.csv")
mndwi_2024 = pd.read_csv("newmndwi_2024.csv")

precip = pd.read_excel("newprecip.xlsx")

lag_df = pd.read_csv("master_with_lag_precipstatic.csv")

# =====================================================
# NORMALIZE ADMIN NAMES
# =====================================================

def normalize(df):

    for c in ["WADMKK","WADMKC"]:
        df[c] = (
            df[c]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace("KOTA ","", regex=False)
            .str.replace("KABUPATEN ","", regex=False)
        )

    return df


datasets = [
    topo, soil, twi, drain,
    ndbi, ibi,
    ndvi_2019, ndvi_2024,
    mndwi_2019, mndwi_2024,
    precip, lag_df
]

for d in datasets:
    normalize(d)

# =====================================================
# STATIC DATASET
# =====================================================

static_df = topo.merge(
    soil[['WADMKK','WADMKC','soil_class_majority']],
    on=['WADMKK','WADMKC'], how='left'
)

static_df = static_df.merge(
    twi[['WADMKK','WADMKC','twi_final']],
    on=['WADMKK','WADMKC'], how='left'
)

static_df = static_df.merge(
    drain[['WADMKK','WADMKC','drainage_pct']],
    on=['WADMKK','WADMKC'], how='left'
)

static_df = static_df.rename(columns={
    "soil_class_majority":"soil_type"
})

static_df["soil_type"] = pd.to_numeric(static_df["soil_type"], errors="coerce").fillna(0)

# =====================================================
# KD TREE
# =====================================================

coords = static_df[['LONG','LAT']].values
tree = KDTree(coords)

RADIUS_KM = 20
DEG_PER_KM = 1/111.32
RADIUS = RADIUS_KM * DEG_PER_KM

# =====================================================
# PRECIP LOOKUP
# =====================================================

precip_lookup = precip.set_index(['WADMKK','WADMKC'])

# =====================================================
# BUILD LOOKUP TABLES (FAST)
# =====================================================

def build_lookup(df, col):

    lookup = {}

    for _,r in df.iterrows():
        lookup[(r["WADMKK"], r["WADMKC"])] = r

    return lookup


ndvi_lookup_2019 = build_lookup(ndvi_2019, None)
ndvi_lookup_2024 = build_lookup(ndvi_2024, None)

mndwi_lookup_2019 = build_lookup(mndwi_2019, None)
mndwi_lookup_2024 = build_lookup(mndwi_2024, None)

ndbi_lookup = build_lookup(ndbi, None)
ibi_lookup = build_lookup(ibi, None)

lag_lookup = lag_df.set_index(["WADMKK","WADMKC"])

# =====================================================
# BUILD DATASET
# =====================================================

def build_dataset(date, ndvi_lookup, mndwi_lookup, year):

    rows = []

    d0 = date.strftime('%Y-%m-%d')
    d1 = (date - timedelta(days=1)).strftime('%Y-%m-%d')
    d2 = (date - timedelta(days=2)).strftime('%Y-%m-%d')

    month_col = date.strftime('%Y-%m')
    year_col = str(year)

    for i,row in static_df.iterrows():

        wadmkk = row['WADMKK']
        wadmkc = row['WADMKC']

        key = (wadmkk, wadmkc)

        # =====================================================
        # PRECIP
        # =====================================================

        p0 = p1 = p2 = precip7 = 0.0

        if key in precip_lookup.index:

            p_row = precip_lookup.loc[key]

            if isinstance(p_row, pd.DataFrame):
                p_row = p_row.iloc[0]

            if d0 in precip_lookup.columns:
                p0 = float(p_row[d0])

            if d1 in precip_lookup.columns:
                p1 = float(p_row[d1])

            if d2 in precip_lookup.columns:
                p2 = float(p_row[d2])

            for j in range(7):

                dj = (date - timedelta(days=j)).strftime('%Y-%m-%d')

                if dj in precip_lookup.columns:
                    precip7 += float(p_row[dj])

        precip3 = p0 + p1 + p2

        # =====================================================
        # NDVI
        # =====================================================

        ndvi_val = 0

        if key in ndvi_lookup:

            r = ndvi_lookup[key]

            if month_col in r.index:
                ndvi_val = r[month_col]

        # =====================================================
        # MNDWI
        # =====================================================

        mndwi_val = 0

        if key in mndwi_lookup:

            r = mndwi_lookup[key]

            if month_col in r.index:
                mndwi_val = r[month_col]

        # =====================================================
        # NDBI
        # =====================================================

        ndbi_val = 0

        if key in ndbi_lookup:

            r = ndbi_lookup[key]

            if year_col in r.index:
                ndbi_val = r[year_col]

        # =====================================================
        # IBI
        # =====================================================

        ibi_val = 0

        if key in ibi_lookup:

            r = ibi_lookup[key]

            col = f"IBI_{year}"

            if col in r.index:
                ibi_val = r[col]

        # =====================================================
        # NEIGHBORS
        # =====================================================

        idx = tree.query_ball_point(coords[i], r=RADIUS)
        idx = [j for j in idx if j != i]

        nb_twi_vals = []
        nb_elev_vals = []
        nb_slope_vals = []
        nb_drain_vals = []
        nb_soil_vals = []

        nb_ndvi_vals = []
        nb_mndwi_vals = []
        nb_ndbi_vals = []
        nb_ibi_vals = []

        for j in idx:

            nb = static_df.iloc[j]

            nb_twi_vals.append(nb["twi_final"])
            nb_elev_vals.append(nb["elevation_mean"])
            nb_slope_vals.append(nb["slope_mean"])
            nb_drain_vals.append(nb["drainage_pct"])
            nb_soil_vals.append(nb["soil_type"])

            nb_key = (nb["WADMKK"], nb["WADMKC"])

            if nb_key in ndvi_lookup:

                r = ndvi_lookup[nb_key]

                if month_col in r.index:
                    nb_ndvi_vals.append(r[month_col])

            if nb_key in mndwi_lookup:

                r = mndwi_lookup[nb_key]

                if month_col in r.index:
                    nb_mndwi_vals.append(r[month_col])

            if nb_key in ndbi_lookup:

                r = ndbi_lookup[nb_key]

                if year_col in r.index:
                    nb_ndbi_vals.append(r[year_col])

            if nb_key in ibi_lookup:

                r = ibi_lookup[nb_key]

                col = f"IBI_{year}"

                if col in r.index:
                    nb_ibi_vals.append(r[col])

        # mean neighbor values

        def mean_or_self(values, self_val):

            if len(values) > 0:
                return np.mean(values)
            return self_val

        nb_twi = mean_or_self(nb_twi_vals, row["twi_final"])
        nb_elev = mean_or_self(nb_elev_vals, row["elevation_mean"])
        nb_slope = mean_or_self(nb_slope_vals, row["slope_mean"])
        nb_drain = mean_or_self(nb_drain_vals, row["drainage_pct"])
        nb_soil = mean_or_self(nb_soil_vals, row["soil_type"])

        nb_ndvi = mean_or_self(nb_ndvi_vals, ndvi_val)
        nb_mndwi = mean_or_self(nb_mndwi_vals, mndwi_val)
        nb_ndbi = mean_or_self(nb_ndbi_vals, ndbi_val)
        nb_ibi = mean_or_self(nb_ibi_vals, ibi_val)

        # =====================================================
        # LAG
        # =====================================================

        lag1 = lag3 = lag7 = 0

        if key in lag_lookup.index:

            lag_row = lag_lookup.loc[key]

            lag1 = lag_row["1daylag"]
            lag3 = lag_row["3daylag"]
            lag7 = lag_row["7daylag"]

        if lag1 == 0:
            lag1 = p0

        if lag3 == 0:
            lag3 = precip3

        if lag7 == 0:
            lag7 = precip7

        # =====================================================
        # SAVE ROW
        # =====================================================

        rows.append({

            "WADMKK": wadmkk.title(),
            "WADMKC": wadmkc.title(),

            "Month": date.month,

            "twi_final": row["twi_final"],
            "elevation_mean": row["elevation_mean"],
            "slope_mean": row["slope_mean"],
            "soil_type": row["soil_type"],
            "drainage_pct": row["drainage_pct"],

            "precip_t0": p0,
            "precip_t1": p1,
            "precip_t2": p2,
            "precip_3d": precip3,
            "precip_7d": precip7,

            "ndvi": ndvi_val,
            "mndwi": mndwi_val,
            "ndbi": ndbi_val,

            "nb_precip_t0": p0,
            "nb_precip_t1": p1,
            "nb_precip_t2": p2,

            "nb_twi": nb_twi,
            "nb_elevation": nb_elev,
            "nb_slope": nb_slope,
            "nb_drainage": nb_drain,
            "nb_soil": nb_soil,

            "nb_ndvi": nb_ndvi,
            "nb_mndwi": nb_mndwi,
            "nb_ndbi": nb_ndbi,

            "1daylag": lag1,
            "3daylag": lag3,
            "7daylag": lag7,

            "ibi": ibi_val,
            "nb_ibi": nb_ibi
        })

    return pd.DataFrame(rows)

# =====================================================
# BUILD DATASETS
# =====================================================

print("Building 2019 dataset...")
df2019 = build_dataset(datetime(2019,12,31), ndvi_lookup_2019, mndwi_lookup_2019, 2019)

print("Building 2024 dataset...")
df2024 = build_dataset(datetime(2024,12,31), ndvi_lookup_2024, mndwi_lookup_2024, 2024)

# =====================================================
# SAVE
# =====================================================

df2019.to_csv("prediction_dataset_2019.csv", index=False)
df2024.to_csv("prediction_dataset_2024.csv", index=False)

print("Finished.")