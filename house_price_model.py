import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print("Eğitim verisi boyutu:", train_data.shape)
print("Test verisi boyutu:", test_data.shape)

# 1. Hedef değişken
y = train_data['SalePrice']

# 2. Giriş verileri (X) için sadeleştirilmiş bir özellik listesi
selected_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                     'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'MSZoning', 'KitchenQual']

X = train_data[selected_features]

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      train_size=0.8, 
                                                      test_size=0.2, 
                                                      random_state=0)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sayısal ve kategorik sütunları ayır
numerical_cols = [cname for cname in X_train.columns if 
                  X_train[cname].dtype in ['int64', 'float64']]

categorical_cols = [cname for cname in X_train.columns if 
                    X_train[cname].dtype == "object"]

# Sayısal veri işleme
numerical_transformer = SimpleImputer(strategy='mean')

# Kategorik veri işleme
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor: Sayısal ve kategorik işlemleri birleştir
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Modeli oluştur (örnek parametrelerle)
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# Pipeline: preprocess + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Pipeline'ı eğit
pipeline.fit(X_train, y_train)

# Doğrulama verisi üzerinde tahmin yap
preds = pipeline.predict(X_valid)

# MAE hesapla
mae = mean_absolute_error(y_valid, preds)
print(f"Validation MAE: {mae}")
