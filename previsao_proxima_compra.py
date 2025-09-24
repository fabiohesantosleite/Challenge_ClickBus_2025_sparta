import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
import matplotlib.pyplot as plt
import joblib

DF = "df_t.csv"
df = pd.read_csv(DF, sep=',')

df['purchase_ts'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])
df = df.sort_values(['fk_contact', 'purchase_ts'])

WINDOW = 30
rows = []
for cust, g in df.groupby('fk_contact'):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        cur = g.loc[i]
        nxt = g.loc[i + 1]
        delta_days = (nxt['purchase_ts'] - cur['purchase_ts']).days
        label = 1 if delta_days <= WINDOW else 0

        row = {
            'customer_id': cust,
            'purchase_ts': cur['purchase_ts'],
            'hour': cur['purchase_ts'].hour,
            'weekday': cur['purchase_ts'].weekday(),
            'gmv': cur['gmv_success'],
            'tickets': cur['total_tickets_quantity_success'],
            'origin': cur['place_origin_departure'],
            'dest': cur['place_destination_departure'],
            'label': label
        }
        rows.append(row)

df_train = pd.DataFrame(rows)


for c in ['origin', 'dest']:
    df_train[c] = df_train[c].fillna('UNK')
    df_train[c + '_enc'] = df_train[c].astype('category').cat.codes

X = df_train[['hour', 'weekday', 'gmv', 'tickets', 'origin_enc', 'dest_enc']].fillna(0)
y = df_train['label']

model = LogisticRegression(solver='liblinear')
model.fit(X, y)
df_train['prob'] = model.predict_proba(X)[:, 1]
df_train.to_csv("next_purchase_predictions.csv", index=False)

fpr, tpr, _ = roc_curve(y, df_train['prob'])
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr);
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC - Próxima Compra (demo)')
plt.savefig('roc_curve.png', dpi=150)

joblib.dump(model, 'model_next_purchase.pkl')
print("Gerado: next_purchase_predictions.csv, roc_curve.png, model_next_purchase.pkl")
print("\nClassification report:\n", classification_report(y, (df_train['prob'] > 0.5).astype(int)))