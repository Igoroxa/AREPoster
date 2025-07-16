import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# --------------------------------------------
# STEP 1: Load your earthquake data
# --------------------------------------------
print("Loading earthquake CSV...")
df = pd.read_csv("usgs_quakes_full1.csv")
print(f"Loaded {df.shape[0]} rows")

# Make sure longitude and latitude are numeric
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df = df.dropna(subset=['longitude', 'latitude'])

print(f"Rows after dropping invalid coordinates: {df.shape[0]}")

# Convert to GeoDataFrame
gdf_quakes = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs="EPSG:4326"
)
print("✅ Earthquake GeoDataFrame created.")

# --------------------------------------------
# STEP 2: Load PB2002 boundaries shapefile
# --------------------------------------------
print("Loading PB2002 boundaries shapefile...")
plates = gpd.read_file("PB2002/PB2002_boundaries.shp")
print("Columns:", plates.columns)

# Filter for subduction only
subduction = plates[plates['Type'].str.lower() == 'subduction']
print(f"Number of subduction lines: {len(subduction)}")

# --------------------------------------------
# STEP 3: Buffer the subduction lines to make them areas
# --------------------------------------------
# Buffer distance in degrees — adjust for your resolution!
BUFFER_DEGREES = 0.5

# Reproject if needed (should already be EPSG:4326)
subduction = subduction.to_crs(gdf_quakes.crs)

# Create buffered polygons around lines
subduction_buffered = subduction.buffer(BUFFER_DEGREES)
print(f"Buffered subduction zones created with {BUFFER_DEGREES} degree buffer.")

# Union all buffered polygons into one big MultiPolygon
subduction_union = subduction_buffered.unary_union

# --------------------------------------------
# STEP 4: Flag earthquakes inside buffered subduction zones
# --------------------------------------------
print("Running point-in-polygon test using buffer...")
gdf_quakes['subduction_flag'] = gdf_quakes.geometry.within(subduction_union).astype(int)

print("\nValue counts for subduction_flag:")
print(gdf_quakes['subduction_flag'].value_counts())

# --------------------------------------------
# STEP 5: Add time features and save final dataset
# --------------------------------------------
gdf_quakes['time'] = pd.to_datetime(gdf_quakes['time'])
gdf_quakes['year'] = gdf_quakes['time'].dt.year
gdf_quakes['month'] = gdf_quakes['time'].dt.month
gdf_quakes['day_of_year'] = gdf_quakes['time'].dt.dayofyear

cols_to_save = [
    'latitude', 'longitude', 'depth', 'mag',
    'year', 'month', 'day_of_year', 'subduction_flag'
]

# Save it!
gdf_quakes[cols_to_save].to_csv("earthquakes_subduction.csv", index=False)
print("✅ Done! Final labeled file saved as earthquakes_subduction.csv.")
