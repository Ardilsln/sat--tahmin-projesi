from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
import joblib


app = FastAPI(
    title="Satış Tahmini API",
    description="Northwind verisi ile satış tahmini sağlar",
    version="1.0"
)

# ------------------------------
# Model ve veri yüklemesi
# ------------------------------
model = joblib.load("sales_forecast_model.pkl")
sales_df = pd.read_csv("features.csv")

le_month = LabelEncoder().fit(sales_df["month"])
le_category = LabelEncoder().fit(sales_df["category_name"])
le_country = LabelEncoder().fit(sales_df["country"])
le_customer = LabelEncoder().fit(sales_df["customer_desc"])
le_product = LabelEncoder().fit(sales_df["product_name"])
# ------------------------------
# Tahmin request modeli
# ------------------------------

class PredictRequest(BaseModel):
    month: str = Field(..., example="2024-01")
    category_name: str
    country: str
    customer_desc: str
    unit_price_x: float = Field(..., gt=0)
    product_name: str
    total_sales: float = Field(..., ge=0)

    @field_validator("month")
    @classmethod
    def month_format_control(cls, v):
        try:
            pd.Period(v, freq="M")
            return v
        except Exception:
            raise ValueError("Ay formatı 'YYYY-MM' şeklinde olmalı.")

# ------------------------------
# /products endpoint
# ------------------------------
@app.get("/products")
def get_products():
    return sales_df["product_name"].unique().tolist()

# ------------------------------
# /sales_summary endpoint
# ------------------------------
@app.get("/sales_summary")
def get_sales_summary():
    summary = sales_df.groupby("product_name")["total_sales"].sum().reset_index()
    return summary.to_dict(orient="records")

# ------------------------------
# /predict endpoint
# ------------------------------
@app.post("/predict")
def predict_sales(req: PredictRequest):
    try:
        # Geçerli label'lar dışındaki verileri kontrol et
        if req.month not in le_month.classes_:
            raise HTTPException(status_code=400, detail=f"Ay '{req.month}' geçerli değil. Kullanılabilir aylar: {list(le_month.classes_)}")
        if req.category_name not in le_category.classes_:
            raise HTTPException(status_code=400, detail=f"Kategori '{req.category_name}' geçerli değil.")
        if req.country not in le_country.classes_:
            raise HTTPException(status_code=400, detail=f"Ülke '{req.country}' geçerli değil.")
        if req.customer_desc not in le_customer.classes_:
            raise HTTPException(status_code=400, detail=f"Müşteri segmenti '{req.customer_desc}' geçerli değil.")
        if req.product_name not in le_product.classes_:
            raise HTTPException(status_code=400, detail=f"Ürün '{req.product_name}' geçerli değil.")

        # Giriş verisini dataframe'e dönüştür
        input_df = pd.DataFrame([{
            "month": req.month,
            "category_name": req.category_name,
            "country": req.country,
            "customer_desc": req.customer_desc,
            "unit_price_x": req.unit_price_x,
            "product_name": req.product_name,
            "total_sales": req.total_sales
        }])

        # Label encoding (fit edilmiş encoder'larla)
        input_df["month"] = le_month.transform(input_df["month"])
        input_df["category_name"] = le_category.transform(input_df["category_name"])
        input_df["country"] = le_country.transform(input_df["country"])
        input_df["customer_desc"] = le_customer.transform(input_df["customer_desc"])
        input_df["product_name"] = le_product.transform(input_df["product_name"])

        prediction = model.predict(input_df)[0]

        return {
            "predicted_quantity": round(prediction, 2),
            "product_name": req.product_name
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# ------------------------------
# /retrain endpoint
# ------------------------------
@app.post("/retrain")
def retrain_model():
    try:
        df = pd.read_csv("features.csv")

            # LabelEncoder yeniden fit ediliyor
        le_month = LabelEncoder().fit(df["month"])
        le_category = LabelEncoder().fit(df["category_name"])
        le_country = LabelEncoder().fit(df["country"])
        le_customer = LabelEncoder().fit(df["customer_desc"])
        le_product = LabelEncoder().fit(df["product_name"])

        # Encode işlemi
        df["month"] = le_month.transform(df["month"])
        df["category_name"] = le_category.transform(df["category_name"])
        df["country"] = le_country.transform(df["country"])
        df["customer_desc"] = le_customer.transform(df["customer_desc"])
        df["product_name"] = le_product.transform(df["product_name"])

        # Özellik ve hedef sütunları
        features = ["month", "category_name", "country", "customer_desc", "unit_price_x", "product_name", "total_sales"]
        X = df[features]
        y = df["quantity"]

        # Eğitim
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        new_model = DecisionTreeRegressor(random_state=42)
        new_model.fit(X_train, y_train)

        # Kaydet
        joblib.dump(new_model, "sales_forecast_model.pkl")

        global model
        model = new_model

        return {"message": "✅ Model yeniden eğitildi ve güncellendi."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

