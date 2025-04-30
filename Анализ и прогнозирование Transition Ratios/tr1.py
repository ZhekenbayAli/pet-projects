import warnings
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
from statsmodels.stats.diagnostic import normal_ad, het_white
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# Подавление специфических предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Загрузка данных
file_path = 'ссылка_на_файл'
data = pd.read_excel(file_path, sheet_name='Sheet2')

# Фильтрация данных по дате
historical_data = data[(data['Дата'] >= '2018Q1') & (data['Дата'] <= '2022Q4')]

# Разделение данных на обучающую и тестовую выборки
train_data = historical_data[(historical_data['Дата'] >= '2018Q1') & (historical_data['Дата'] <= '2021Q2')]
test_data = historical_data[(historical_data['Дата'] >= '2018Q3') & (historical_data['Дата'] <= '2022Q4')]

# Определение независимых переменных
independent_variables = [
    col for col in data.columns
    if col not in ['Дата', 'pk_2>2']
]

# Генерация комбинаций независимых переменных
combinations = list(itertools.combinations(independent_variables, 3))

def remove_constant_columns(df):
    """Удаление столбцов с постоянными значениями."""
    return df.loc[:, (df != df.iloc[0]).any()]

def calculate_max_vif(X):
    """Расчет максимального коэффициента VIF."""
    return max([variance_inflation_factor(X.values, i) for i in range(X.shape[1])])

def test_model(X, y, model):
    """Диагностика регрессионной модели."""
    y_pred = model.predict(X)
    residuals = y - y_pred

    diagnostics = {
        "Adjusted R-squared": model.rsquared_adj,
        "p-values": model.pvalues.max(),
        "F-value": model.f_pvalue,
        "VIF": calculate_max_vif(X),
        "Durbin-Watson": durbin_watson(residuals),
        "Breusch-Pagan p-value": het_breuschpagan(residuals, X)[1],
        "White Test p-value": het_white(residuals, X)[1],
        "Normality Anderson-Darling p-value": normal_ad(residuals)[1],
        "Normality KS p-value": stats.kstest(residuals, 'norm')[1],
    }

    # Тест причинности Грейнджера
    for i in range(X.shape[1]):
        if X.iloc[:, i].nunique() > 1:
            granger_result = grangercausalitytests(
                pd.concat([y, X.iloc[:, [i]]], axis=1), maxlag=[2], verbose=False
            )
            diagnostics["Granger Causality p-value"] = min(
                result[0]["ssr_chi2test"][1] for _, result in granger_result.items()
            )
    return diagnostics

def filter_model_results(results):
    """Применение фильтров для оценки качества модели."""
    return (
        0.25 <= results["Adjusted R-squared"] <= 0.95
        and results["p-values"] <= 0.12
        and results["F-value"] <= 0.12
        and results["VIF"] <= 10
        and (0.773 <= results["Durbin-Watson"] <= 1.411 or results["Durbin-Watson"] >= 1.411)
    )

# Анализ моделей
model_results = []

for combo in combinations:
    print(f"Обрабатывается комбинация: {combo}")
    
    # Подготовка обучающих данных
    X_train = add_constant(train_data[list(combo)].fillna(train_data.median()))
    y_train = train_data['pk_2>2']
    
    # Построение модели
    model = sm.OLS(y_train, X_train).fit()

    # Подготовка тестовых данных
    X_test = add_constant(test_data[list(combo)].fillna(test_data.median()))
    y_test = test_data['pk_2>2']

    # Оценка модели
    test_results = test_model(X_test, y_test, model)

    # Сохранение результатов, если модель удовлетворяет критериям
    if filter_model_results(test_results):
        model_results.append({
            "Variables": combo,
            "Equation": f"y = {model.params[0]:.4f} + " + " + ".join(
                [f"{coef:.4f}*{col}" for coef, col in zip(model.params[1:], X_train.columns[1:])]
            ),
            **test_results
        })

# Сохранение результатов в Excel
df_results = pd.DataFrame(model_results)
df_results.to_excel("models_tr7_1.xlsx", index=False)
print("Модели и диагностика сохранены в файл models_tr7_1.xlsx")
