import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
DATASET_PATH = './Housing.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 1
IS_SHUFFLE = True

# Load and preprocess data
df = pd.read_csv(DATASET_PATH)
print(df)
df.info()

# Categorical encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

for col_name in categorical_cols:
    n_categories = df[col_name].nunique()
    print(f'Number of categories in {col_name}: {n_categories}')

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(df[categorical_cols])
encoded_categorical_df = pd.DataFrame(
    encoded_categorical_cols, columns=categorical_cols)
numerical_df = df.drop(categorical_cols, axis=1)
encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)

print(encoded_df)

# Normalization
normalizer = StandardScaler()
dataset_arr = normalizer.fit_transform(encoded_df)

# Train-test split
X, y = dataset_arr[:, 1:], dataset_arr[:, 0]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=IS_SHUFFLE
)

print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of val samples: {X_val.shape[0]}')

# Helper function for model evaluation and plotting


def evaluate_and_plot(regressor, X_train, y_train, X_val, y_val, title):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    print('Evaluation results on validation set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    plt.figure()
    plt.scatter(X_val[:, 0], y_val, s=20,
                edgecolor="black", c="green", label="True")
    plt.scatter(X_val[:, 0], y_pred, s=20, edgecolor="black",
                c="darkorange", label="Prediction")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.show()


# Random Forest
rf_regressor = RandomForestRegressor(random_state=RANDOM_STATE)
evaluate_and_plot(rf_regressor, X_train, y_train, X_val,
                  y_val, "Random Forest Regression")

# AdaBoost
ada_regressor = AdaBoostRegressor(random_state=RANDOM_STATE)
evaluate_and_plot(ada_regressor, X_train, y_train,
                  X_val, y_val, "AdaBoost Regression")

# Gradient Boosting
gb_regressor = GradientBoostingRegressor(random_state=RANDOM_STATE)
evaluate_and_plot(gb_regressor, X_train, y_train, X_val,
                  y_val, "Gradient Boosting Regression")
