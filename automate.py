import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
import xgboost as xgb

# Define API key
together_api_key = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"

# Define AI agent class
class AgentAI:
    def __init__(self):
        self.models = {
            'Regression': {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression': LinearRegression(),
                'Lasso Regression': Lasso(),
                'Ridge Regression': Ridge(),
                'Elastic Net Regression': ElasticNet(),
                'Decision Tree Regression': DecisionTreeRegressor(),
                'Random Forest Regression': RandomForestRegressor(),
                'Extra Trees Regressor': ExtraTreesRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGBoost Regressor': xgb.XGBRegressor(),
                'KNN Regressor': KNeighborsRegressor(),
                'SVR': SVR()
            },
            'Classification': {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Extra Trees Classifier': ExtraTreesClassifier(),
                'Gradient Boosting Classifier': GradientBoostingClassifier(),
                'XGBoost Classifier': xgb.XGBClassifier(),
                'KNN Classifier': KNeighborsClassifier(),
                'SVC': SVC()
            },
            'Multi-Class Classification': {
                'Multinomial NB': MultinomialNB(),
                'Bernoulli NB': BernoulliNB(),
                'Complement NB': ComplementNB()
            }
        }
    }

    def train_model(self, X_train, y_train, task, model_name, degree=2):
        if task == 'Regression':
            if model_name == 'Polynomial Regression':
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(X_train)
                self.models[task][model_name].fit(X_train_poly, y_train)
            else:
                self.models[task][model_name].fit(X_train, y_train)
        else:
            self.models[task][model_name].fit(X_train, y_train)

    def predict(self, X_test, task, model_name, degree=2):
        if task == 'Regression' and model_name == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree=degree)
            X_test_poly = poly_features.fit_transform(X_test)
            return self.models[task][model_name].predict(X_test_poly)
        else:
            return self.models[task][model_name].predict(X_test)

    def evaluate(self, y_test, y_pred, task):
        if task == 'Regression':
            return mean_squared_error(y_test, y_pred)
        else:
            return accuracy_score(y_test, y_pred)

# Create an instance of the AI agent
agent = AgentAI()

# Streamlit app
st.title("AI Agent for Machine Learning Tasks")

task = st.selectbox("Select Task", ["Regression", "Classification", "Multi-Class Classification"])
model_name = st.selectbox("Select Model", list(agent.models[task].keys()))

degree = st.slider("Degree for Polynomial Regression", min_value=2, max_value=3, value=2)

test_size = st.slider("Test Size", min_value=0.1, max_value=0.3, step=0.05, value=0.2)

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    agent.train_model(X_train, y_train, task, model_name, degree=degree)
    y_pred = agent.predict(X_test, task, model_name, degree=degree)
    metric = agent.evaluate(y_test, y_pred, task)

    st.write(f"Model: {model_name}")
    st.write(f"Metric: {metric:.2f}")
    st.write(f"API Key: {together_api_key}")

    # Find the best model
    best_model = None
    best_metric = float('inf') if task == 'Regression' else 0
    for model in agent.models[task]:
        agent.train_model(X_train, y_train, task, model)
        y_pred = agent.predict(X_test, task, model)
        metric = agent.evaluate(y_test, y_pred, task)
        if task == 'Regression' and metric < best_metric:
            best_model = model
            best_metric = metric
        elif task != 'Regression' and metric > best_metric:
            best_model = model
            best_metric = metric

    st.write(f"Best Model: {best_model}")
    st.write(f"Best Metric: {best_metric:.2f}")
