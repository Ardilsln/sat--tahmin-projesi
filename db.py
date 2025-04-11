from sqlalchemy import create_engine, text
import psycopg2
import sys
import locale
import pandas as pd
import numpy as np

print("System Information:")
print(f"Default Encoding: {sys.getdefaultencoding()}")
print(f"Filesystem Encoding: {sys.getfilesystemencoding()}")
print(f"Locale Encoding: {locale.getpreferredencoding()}")

def get_connection_string():
    params = {
        'database': 'GYK2Northwind',
        'user': 'postgres',
        'password': '1234',  # ← burada doğru şifre olduğundan emin ol
        'host': 'localhost',
        'port': '5432'
    }
    conn_string = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
    return conn_string

try:
    engine = create_engine(get_connection_string())
    
    with engine.connect() as conn:
        
        result = conn.execute(text('SELECT * FROM "products" LIMIT 5;'))

        print("\n Ürün Verileri (İlk 5):\n")
        for row in result:
            try:
                clean_row = []
                for item in row:
                    # Her sütunu UTF-8'e güvenli çeviriyoruz
                    clean_item = str(item).encode('utf-8', errors='replace').decode('utf-8')
                    clean_row.append(clean_item)
                print(" | ".join(clean_row))
            except Exception as e:
                print("⚠️ Satır yazdırılırken hata:", e)
                print("Ham veri:", repr(row))
        #  Gerekli tabloları Pandas ile çekiyoruz
        categories = pd.read_sql('SELECT * FROM "categories";', conn)
        customer_customer_demo = pd.read_sql('SELECT * FROM "customer_customer_demo";', conn)
        customer_demographics = pd.read_sql('SELECT * FROM "customer_demographics";', conn)
        customers = pd.read_sql('SELECT * FROM "customers";', conn)
        employee_territories = pd.read_sql('SELECT * FROM "employee_territories";', conn)
        employees = pd.read_sql('SELECT * FROM "employees";', conn)
        order_details = pd.read_sql('SELECT * FROM "order_details";', conn)
        orders = pd.read_sql('SELECT * FROM "orders";', conn)
        products = pd.read_sql('SELECT * FROM "products";', conn)
        region = pd.read_sql('SELECT * FROM "region";', conn)
        shippers = pd.read_sql('SELECT * FROM "shippers";', conn)
        suppliers = pd.read_sql('SELECT * FROM "suppliers";', conn)
        territories = pd.read_sql('SELECT * FROM "territories";', conn)
        us_states = pd.read_sql('SELECT * FROM "us_states";', conn)


        print("\n Eksik Veri Kontrolü:\n")
        for name, df in [('orders', orders), ('order_details', order_details), ('products', products),
                         ('categories', categories), ('customer_customer_demo' , customer_customer_demo),('customer_demographics' , customer_demographics),
                          ('customers' , customers), ('employee_territories' , employee_territories),
                           ('employees' , employees), ('region' , region), ('shippers' , shippers),
                           ('suppliers' , suppliers), ('territories' , territories), ('us_states' , us_states)]:
            print(f"\n{name.upper()} tablosu eksik veri:\n{df.isnull().sum()}")

        print("\n Tabloları birleştiriyoruz...")
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        sales = pd.merge(order_details, orders, on='order_id')
        sales = pd.merge(sales, products, on='product_id', how='left')
        sales = pd.merge(sales, categories, on='category_id', how='left')
        sales = pd.merge(sales, customers, on='customer_id', how='left')
        sales = pd.merge(sales, customer_customer_demo, on='customer_id', how='left')
        sales = pd.merge(sales, customer_demographics, on='customer_type_id', how='left')
        
        sales['order_date'] = pd.to_datetime(sales['order_date'], errors='coerce')
        sales['month'] = sales['order_date'].dt.to_period("M").astype(str)
        
        print("\n📈 Sayısal kolonlar için korelasyon matrisi:\n")
        corr = sales.corr(numeric_only=True)
        print(corr)

        # ⚠️ Aykırı Değer Kontrolü
        q1_q = sales['quantity'].quantile(0.25)
        q3_q = sales['quantity'].quantile(0.75)
        iqr_q = q3_q - q1_q
        outliers_q = sales[(sales['quantity'] < q1_q - 1.5 * iqr_q) | (sales['quantity'] > q3_q + 1.5 * iqr_q)]

        q1_p = sales['unit_price_x'].quantile(0.25)
        q3_p = sales['unit_price_x'].quantile(0.75)
        iqr_p = q3_p - q1_p
        outliers_p = sales[(sales['unit_price_x'] < q1_p - 1.5 * iqr_p) | (sales['unit_price_x'] > q3_p + 1.5 * iqr_p)]

        print(f"\n⚠️ Aykırı quantity değeri sayısı: {len(outliers_q)}")
        print(f"⚠️ Aykırı unit_price değeri sayısı: {len(outliers_p)}")

        # 👀 Örnek veri
        print("\n sales tablosu (ilk 5 satır):")
        print(sales[['order_id', 'product_id', 'quantity', 'unit_price_x', 'month', 'customer_desc']].head())
        
        # 🔧 Yeni Özellikler

        sales['total_sales'] = sales['unit_price_x'] * sales['quantity']
        # Eksik veri doldurma (önlem amaçlı)
        sales['customer_desc'] = sales['customer_desc'].fillna('Unknown')
        sales['category_name'] = sales['category_name'].fillna('Unknown')
        sales['country'] = sales['country'].fillna('Unknown')
        sales['product_name'] = sales['product_name'].fillna('Unknown')


except Exception as e:
    print("error: " + str(e))
    # print("\n❌ Detailed Error: " + str(e))
    print(f"Error Type: {type(e)}")
    if isinstance(e, psycopg2.Error):
        print(f"psycopg2 Error Details: {e}")



sales.to_csv("features.csv", index=False)

# -----------------------
#  B Şıkkı: Model Eğitimi 
# -----------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib

# Hazırlıklar
sales['customer_desc'] = sales['customer_desc'].fillna('Unknown')
sales['category_name'] = sales['category_name'].fillna('Unknown')
sales['country'] = sales['country'].fillna('Unknown')

# Özellikler ve hedef
features = ['month', 'category_name', 'country', 'customer_desc', 'unit_price_x', 'product_name', 'total_sales']

X = sales[features]
y = sales['quantity']

# Encoding
le_month = LabelEncoder()
le_category = LabelEncoder()
le_country = LabelEncoder()
le_segment = LabelEncoder()

X.loc[:, 'month'] = LabelEncoder().fit_transform(X['month'])
X.loc[:, 'category_name'] = LabelEncoder().fit_transform(X['category_name'])
X.loc[:, 'country'] = LabelEncoder().fit_transform(X['country'])
X.loc[:, 'customer_desc'] = LabelEncoder().fit_transform(X['customer_desc'])
X.loc[:, 'product_name'] = LabelEncoder().fit_transform(X['product_name'])



# Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Eğitimi
model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# Değerlendirme
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n Model başarıyla eğitildi!")
print(f" R2 Skoru: {r2:.4f}")
print(f" RMSE: {rmse:.2f}")

# Kaydet
joblib.dump(model, "sales_forecast_model.pkl")
print(" Model kaydedildi.")



