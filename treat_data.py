import pandas as pd
import numpy as np

CSV_IN = "df_t.csv"
CSV_OUT = "clickbus_clean.csv"

df = pd.read_csv(CSV_IN, sep=",")
print("Colunas encontradas no dataset:\n", df.columns.tolist(), "\n")


if "date_purchase" in df.columns:
    if "time_purchase" in df.columns:
        df["time_purchase"] = df["time_purchase"].fillna("00:00:00")
        df["purchase_ts"] = pd.to_datetime(
            df["date_purchase"].astype(str) + " " + df["time_purchase"].astype(str),
            errors="coerce"
        )
    else:
        df["purchase_ts"] = pd.to_datetime(df["date_purchase"], errors="coerce")
elif "purchase_date" in df.columns:  # fallback se vier com outro nome
    df["purchase_ts"] = pd.to_datetime(df["purchase_date"], errors="coerce")
else:
    raise ValueError("Nenhuma coluna de data de compra encontrada no CSV!")


for c in ["place_origin_return", "place_destination_return"]:
    if c in df.columns:
        df[c] = df[c].replace("0", pd.NA)


for c in [
    "place_origin_departure", "place_destination_departure",
    "place_origin_return", "place_destination_return",
    "fk_departure_ota_bus_company", "fk_return_ota_bus_company"
]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.upper().replace("NAN", pd.NA)


if "place_origin_departure" in df.columns and "place_destination_departure" in df.columns:
    df["route"] = (
        df["place_origin_departure"].astype(str).str.strip().str.upper()
        + " - " +
        df["place_destination_departure"].astype(str).str.strip().str.upper()
    )
else:
    df["route"] = "UNKNOWN"


if "gmv_success" in df.columns:
    df["gmv_success"] = pd.to_numeric(df["gmv_success"], errors="coerce").fillna(0.0)

if "total_tickets_quantity_success" in df.columns:
    df["total_tickets_quantity_success"] = (
        pd.to_numeric(df["total_tickets_quantity_success"], errors="coerce")
        .fillna(0)
        .astype(int)
    )


before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Removidas {before - after} linhas duplicadas.")


df["week_day"] = df["purchase_ts"].dt.weekday   # 0=segunda, 6=domingo
df["hour_band"] = pd.cut(
    df["purchase_ts"].dt.hour,
    bins=[-1, 6, 12, 18, 24],
    labels=["madrugada","manha","tarde","noite"])

if "place_origin_return" in df.columns:
    df["is_return"] = np.where(df["place_origin_return"].notna(), 1, 0)
else:
    df["is_return"] = 0


rename_map = {
    "nk_ota_localizer_id": "order_id",
    "fk_contact": "customer_id"}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

df.to_csv(CSV_OUT, sep=";", index=False)
print("Arquivo tratado salvo em:", CSV_OUT)

print("\nResumo final:")
print("Linhas:", len(df))
print("Colunas:", len(df.columns))
print("\nValores nulos (%):")
print(df.isna().mean().round(3) * 100)
