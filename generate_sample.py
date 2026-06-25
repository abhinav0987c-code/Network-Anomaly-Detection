import pandas as pd
import joblib
import random

features = joblib.load('feature_names.joblib')
data = []
protocols = ['tcp', 'udp', 'icmp']
services = ['http', 'private', 'domain_u', 'smtp', 'ftp_data']
flags = ['SF', 'S0', 'REJ']

for i in range(50):
    row = []
    for f in features:
        if f == 'protocol_type':
            row.append(random.choice(protocols))
        elif f == 'service':
            row.append(random.choice(services))
        elif f == 'flag':
            row.append(random.choice(flags))
        elif 'rate' in f:
            row.append(round(random.random(), 2))
        elif 'byte' in f:
            row.append(random.randint(0, 5000))
        else:
            row.append(random.randint(0, 10))
    data.append(row)

df = pd.DataFrame(data, columns=features)
df.to_csv('sample_bulk_traffic.csv', index=False)
print("sample_bulk_traffic.csv created successfully with", len(df), "rows.")
