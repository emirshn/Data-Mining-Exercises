import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Preprocessing ---
TARGET_ROWS = 25000 

base_dir = os.path.dirname(__file__)
input_file = os.path.join(base_dir, 'weather.csv')
output_file = os.path.join(base_dir, f'weather_25k.csv')

df = pd.read_csv(input_file, parse_dates=['StartTime(UTC)', 'EndTime(UTC)'])

# Clean data
df.drop(columns=['EventId', 'ZipCode', 'AirportCode', 'TimeZone','LocationLat', 'LocationLng', 'City', 'County'], inplace=True, errors='ignore')
df.dropna(subset=['Type', 'Severity', 'StartTime(UTC)', 'EndTime(UTC)', 'State', 'Precipitation(in)'], inplace=True)

# Create new features from time information
start_time = df['StartTime(UTC)']
end_time = df['EndTime(UTC)']

df['DurationHours'] = (end_time - start_time).dt.total_seconds() / 3600

month = start_time.dt.month
df['Season'] = start_time.dt.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

df['TimeOfDay'] = pd.cut(start_time.dt.hour,
                         bins=[-1, 6, 12, 18, 24],
                         labels=['Night', 'Morning', 'Afternoon', 'Evening'])

df['DayType'] = start_time.dt.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Stratified sampling by Type
grouped = df.groupby('Type')
sampled = grouped.apply(
    lambda x: x.sample(
        n=int(len(x) / len(df) * TARGET_ROWS),
        random_state=42
    )
).reset_index(drop=True)

# Bin mapping functions
def bin_precipitation(row):
    event_type = row['Type']
    precip = row['Precipitation(in)']
    
    if pd.isna(precip) or precip == 0:
        if event_type in ['Rain', 'Snow', 'Hail', 'Storm', 'Other Precipitation']:
            return 'Precipitation=Trace'
        else:
            return 'Precipitation=None'
    
    if precip <= 0.1:
        return 'Precipitation=Low'
    elif precip <= 0.5:
        return 'Precipitation=Medium'
    else:
        return 'Precipitation=High'

def bin_duration(hours):
    if hours < 1:
        return 'Duration=Short'
    elif hours <= 3:
        return 'Duration=Medium'
    else:
        return 'Duration=Long'

def map_region(state):
    regions = {
        'Northeast': {'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'},
        'Midwest': {'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'},
        'South': {'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'MS', 'AL', 'TX', 'OK', 'AR', 'LA'},
        'West': {'MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI'}
    }
    for region, states in regions.items():
        if state in states:
            return region
    return 'Other'

sampled['PrecipitationLevel'] = sampled.apply(bin_precipitation, axis=1)
sampled['DurationCategory'] = sampled['DurationHours'].map(bin_duration)
sampled['Region'] = sampled['State'].map(map_region)

sampled.to_csv(output_file, index=False)
print(f"Saved {len(sampled)} stratified rows to '{output_file}'")

df = pd.read_csv('weather_25k.csv')

# categorical features for clustering
categorical_features = [
    'Type', 'Severity', 'Season', 'Region',
    'PrecipitationLevel', 'DurationCategory',
    'TimeOfDay', 'DayType'
]

df_clustering = df[categorical_features].copy()

# One-hot encode categorical attributes
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df_clustering)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features), index=df_clustering.index)

# --- KMeans ---
kmeans = KMeans(n_clusters=5)
df_clustering['Cluster_KMeans'] = kmeans.fit_predict(encoded_df)

# --- AGNES ---
agnes = AgglomerativeClustering(n_clusters=5)
df_clustering['Cluster_AGNES'] = agnes.fit_predict(encoded_df)

# --- DBSCAN ---
dbscan = DBSCAN(eps=1.5, min_samples=10)
df_clustering['Cluster_DBSCAN'] = dbscan.fit_predict(encoded_df)

# ---- Visualization ---- 
def visualize_clusters(method, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(encoded_df)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette='tab10', s=30, legend='full')
    plt.title(f'{method} Clustering')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_clusters('KMeans', df_clustering['Cluster_KMeans'])
visualize_clusters('AGNES', df_clustering['Cluster_AGNES'])
visualize_clusters('DBSCAN', df_clustering['Cluster_DBSCAN'])

# ---- Evaluation ---- 
def analyze_clusters(label_column):
    print(f"\nNumber of points in each cluster for {label_column}:")
    cluster_sizes = df_clustering[label_column].value_counts().sort_index()
    print(cluster_sizes)

    print(f"\nEvent Type Distribution in {label_column}:")
    event_type_dist = df_clustering.groupby(label_column)['Type'].value_counts(normalize=True)
    print(event_type_dist.map(lambda x: f"{x:.2f}"))

    print(f"\nSeverity Distribution in {label_column}:")
    severity_dist = df_clustering.groupby(label_column)['Severity'].value_counts(normalize=True)
    print(severity_dist.map(lambda x: f"{x:.2f}"))

analyze_clusters('Cluster_KMeans')
analyze_clusters('Cluster_AGNES')
analyze_clusters('Cluster_DBSCAN')
