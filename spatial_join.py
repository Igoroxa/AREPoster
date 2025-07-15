import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------
# STEP 1: Load your earthquake data
# ---------------------------
print("Loading earthquake CSV...")
df = pd.read_csv("usgs_quakes_full1.csv")

print(f"Loaded {df.shape[0]} rows")

# See the columns
print("\nColumns in CSV:", df.columns.tolist())

# Preview rows
print(df.head())

# ---------------------------
# STEP 2: Make sure longitude and latitude are numeric
# ---------------------------
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

# Drop any rows with missing or invalid coordinates
df = df.dropna(subset=['longitude', 'latitude'])

print(f"Rows after dropping bad coordinates: {df.shape[0]}")

# ---------------------------
# STEP 3: Convert earthquakes to GeoDataFrame
# ---------------------------
print("Converting to GeoDataFrame...")
gdf_quakes = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs="EPSG:4326"
)

print(gdf_quakes.head())

# ---------------------------
# STEP 4: Load PB2002 boundaries shapefile
# ---------------------------
print("Loading PB2002 boundaries shapefile...")
plates = gpd.read_file("PB2002/PB2002_boundaries.shp")
print(plates.head())

print("Columns:", plates.columns)
print("Boundary types:", plates['Type'].unique())

subduction = plates[plates['Type'].str.lower() == 'subduction']


print(f"Subduction boundaries: {len(subduction)}")

# ---------------------------
# STEP 6: Flag earthquakes inside subduction zones
# ---------------------------
print("Running point-in-polygon test...")
gdf_quakes['subduction_flag'] = gdf_quakes.geometry.within(subduction.unary_union).astype(int)

print(gdf_quakes[['latitude', 'longitude', 'subduction_flag']].head())

# ---------------------------
# STEP 7: Save final labeled CSV
# ---------------------------
# Add your time features if they’re not already there:
gdf_quakes['time'] = pd.to_datetime(gdf_quakes['time'])
gdf_quakes['year'] = gdf_quakes['time'].dt.year
gdf_quakes['month'] = gdf_quakes['time'].dt.month
gdf_quakes['day_of_year'] = gdf_quakes['time'].dt.dayofyear

cols_to_save = [
    'latitude', 'longitude', 'depth', 'mag',
    'year', 'month', 'day_of_year', 'subduction_flag'
]

print("Saving earthquakes_subduction.csv...")
gdf_quakes[cols_to_save].to_csv("earthquakes_subduction.csv", index=False)
print("✅ Done! Final labeled file saved as earthquakes_subduction.csv.")
