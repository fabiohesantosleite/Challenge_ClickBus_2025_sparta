import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib


DF = "df_t.csv"
df = pd.read_csv(DF, sep=',')

df['purchase_ts'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])

df = df.sort_values(['fk_contact', 'purchase_ts'])

rows = []
for cust, g in df.groupby('fk_contact'):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        cur = g.loc[i]
        nxt = g.loc[i + 1]
        next_route = str(nxt['place_origin_departure']) + ' - ' + str(nxt['place_destination_departure'])

        row = {
            'customer_id': cust,
            'purchase_ts': cur['purchase_ts'],
            'hour': cur['purchase_ts'].hour,
            'weekday': cur['purchase_ts'].weekday(),
            'gmv': cur['gmv_success'],
            'tickets': cur['total_tickets_quantity_success'],
            'last_route': str(cur['place_origin_departure']) + ' - ' + str(cur['place_destination_departure']),
            'next_route': next_route
        }
        rows.append(row)

d = pd.DataFrame(rows)

TOPN = 10
top_routes = d['next_route'].value_counts().nlargest(TOPN).index.tolist()
d['next_route_reduced'] = d['next_route'].where(d['next_route'].isin(top_routes), 'OTHER')

le_route = LabelEncoder()
d['route_label'] = le_route.fit_transform(d['next_route_reduced'])
d['last_route_enc'] = d['last_route'].astype('category').cat.codes

X = d[['hour', 'weekday', 'gmv', 'tickets', 'last_route_enc']].fillna(0)
y = d['route_label']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
pred = clf.predict(X)
d['pred_route'] = le_route.inverse_transform(pred)
d[['customer_id', 'last_route', 'next_route', 'pred_route']].to_csv('trecho_predictions.csv', index=False)

cm = confusion_matrix(y, pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion matrix (demo)')
plt.colorbar()
plt.savefig('conf_matrix.png', dpi=150)

joblib.dump(clf, 'model_next_route.pkl')
print("Gerado: trecho_predictions.csv, conf_matrix.png, model_next_route.pkl")
print("\nClassification report:\n", classification_report(y, pred, zero_division=0))