import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

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

# Convert each row to a transaction
transactions = []
for _, row in df.iterrows():
    items = [
        f"Type={row['Type']}",
        f"Severity={row['Severity']}",
        f"Season={row['Season']}",
        f"Region={row['Region']}",
        f"{row['PrecipitationLevel']}",
        f"{row['DurationCategory']}",
        f"TimeOfDay={row['TimeOfDay']}",
        f"DayType={row['DayType']}"
    ]
    transactions.append(items)

# One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# --- Apriori ---

# Find frequent itemsets 
frequent_apriori = apriori(df_encoded, min_support=0.1, use_colnames=True)

print("Apriori Frequent Itemsets:")
print(frequent_apriori.sort_values(by='support', ascending=False).head(20))
rules_apriori  = association_rules(frequent_apriori, metric="confidence", min_threshold=0.5)
# Filter rules with more than 3 items
rules_apriori = rules_apriori[
    rules_apriori.apply(lambda row: len(row['antecedents'] | row['consequents']) > 3, axis=1)
]

# Sort rules by confidence
rules_apriori  = rules_apriori .sort_values(by='confidence', ascending=False)
print("\nApriori Rules:")
for _, rule in rules_apriori.head(20).iterrows():
    lhs = ', '.join(map(str, rule['antecedents']))
    rhs = ', '.join(map(str, rule['consequents']))
    print(f"{{{lhs}}} → {{{rhs}}} (conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})")

# --- FP-Growth ---

frequent_fpgrowth = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

print("\nFP-Growth Frequent Itemsets:")
print(frequent_fpgrowth.sort_values(by='support', ascending=False).head(20))
rules_fpgrowth = association_rules(frequent_fpgrowth, metric="confidence", min_threshold=0.5)
rules_fpgrowth = rules_apriori[
    rules_apriori.apply(lambda row: len(row['antecedents'] | row['consequents']) > 3, axis=1)
]
# Sort rules by confidence
rules_fpgrowth = rules_fpgrowth.sort_values(by='confidence', ascending=False)
print("\nFP-Growth Rules:")
for _, rule in rules_fpgrowth.head(20).iterrows():
    lhs = ', '.join(rule['antecedents'])
    rhs = ', '.join(rule['consequents'])
    print(f"{{{lhs}}} → {{{rhs}}} (conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})")
