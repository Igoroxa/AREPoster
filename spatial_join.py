import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------
# STEP 1: Load your earthquake data
# ---------------------------
print("Loading earthquake CSV...")
df = pd.read_csv("usgs_quakes_full1.csv")
print(f"Loaded {df.shape[0]} rows")

# ---------------------------
# STEP 2: Convert earthquakes to GeoDataFrame
# ---------------------------
print("Converting to GeoDataFrame...")
gdf_quakes = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

print(gdf_quakes.head())

# ---------------------------
# STEP 3: Load PB2002 boundaries shapefile
# ---------------------------
print("Loading PB2002 boundaries shapefile...")
plates = gpd.read_file("PB2002/PB2002_boundaries.shp")   # note the PB2002 folder!
print(plates.head())

# Confirm boundary types available:
print("Boundary types:", plates['TYPE'].unique())

# ---------------------------
# STEP 4: Filter for subduction zones only
# ---------------------------
subduction = plates[plates['TYPE'].str.upper() == 'SUBDUCTION']

print(f"Subduction boundaries: {len(subduction)}")
subduction.plot(figsize=(10, 6))

# ---------------------------
# STEP 5: Flag earthquakes inside subduction zones
# ---------------------------
print("Running point-in-polygon test...")
gdf_quakes['subduction_flag'] = gdf_quakes.geometry.within(subduction.unary_union).astype(int)

print(gdf_quakes[['latitude', 'longitude', 'subduction_flag']].head())

# ---------------------------
# STEP 6: Save final labeled CSV
# ---------------------------
cols_to_save = [
    'latitude', 'longitude', 'depth', 'mag',
    'year', 'month', 'day_of_year', 'subduction_flag'
]

print("Saving earthquakes_subduction.csv...")
gdf_quakes[cols_to_save].to_csv("earthquakes_subduction.csv", index=False)
print("Done! Final labeled file saved.")
