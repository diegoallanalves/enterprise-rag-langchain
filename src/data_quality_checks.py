import pandas as pd
from config import ORDERS_FILE, INVENTORY_FILE, OUTPUTS_DIR

def run_data_quality_checks():
    orders = pd.read_csv(ORDERS_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)
    df = orders.merge(inventory, on=["Product_ID", "Product_Name"], how="left")

    df["Stock_Shortage"] = df["Quantity_Ordered"] > df["Current_Stock"]
    df["Shortage_Quantity"] = (df["Quantity_Ordered"] - df["Current_Stock"]).clip(lower=0)
    df["Promotion_Supply_Risk"] = (
        df["Next_Month_Promotion"].fillna("No").eq("Yes")
        & df["Next_Month_Supply_Status"].fillna("Normal").eq("Restricted")
    )
    df["Promotion_Shortage_Risk"] = df["Stock_Shortage"] & df["Promotion_Supply_Risk"]

    mismatch = orders.groupby("PO_Number")["Customer_Name"].nunique(dropna=True).reset_index(name="Unique_Names")
    mismatch["Customer_Name_Mismatch"] = mismatch["Unique_Names"] > 1
    df = df.merge(mismatch[["PO_Number", "Customer_Name_Mismatch"]], on="PO_Number", how="left")

    df["Missing_Customer_Name"] = df["Customer_Name"].isna() | (df["Customer_Name"].astype(str).str.strip() == "")

    def risk(row):
        if row["Promotion_Shortage_Risk"]:
            return "High"
        if row["Stock_Shortage"] and row["Customer_Name_Mismatch"]:
            return "High"
        if (
            row["Stock_Shortage"]
            or row["Promotion_Supply_Risk"]
            or row["Customer_Name_Mismatch"]
            or row["Missing_Customer_Name"]
        ):
            return "Medium"
        return "Low"

    def recommendation(row):
        actions = []
        if row["Promotion_Shortage_Risk"]:
            actions.append("Do not approve order until stock is replenished or supply planning confirms allocation")
        elif row["Promotion_Supply_Risk"]:
            actions.append("Review next month promotion supply before confirming order")
        if row["Stock_Shortage"]:
            actions.append("Review inventory before shipment")
        if row["Customer_Name_Mismatch"]:
            actions.append("Standardize customer name")
        if row["Missing_Customer_Name"]:
            actions.append("Request missing customer name")
        return "; ".join(actions) if actions else "No immediate action required"

    df["AI_Risk_Level"] = df.apply(risk, axis=1)
    df["AI_Recommendation"] = df.apply(recommendation, axis=1)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "ai_quality_check_results.csv", index=False)
    return df

def build_quality_summary():
    df = run_data_quality_checks()
    return f"""
High risk records: {len(df[df['AI_Risk_Level'] == 'High'])}
Medium risk records: {len(df[df['AI_Risk_Level'] == 'Medium'])}
Orders exceeding stock: {int(df['Stock_Shortage'].sum())}
Orders with promotion supply risk: {int(df['Promotion_Supply_Risk'].sum())}
Orders with promotional stock shortage risk: {int(df['Promotion_Shortage_Risk'].sum())}
Records with customer name mismatch: {int(df['Customer_Name_Mismatch'].sum())}
Records with missing customer name: {int(df['Missing_Customer_Name'].sum())}
""", df
