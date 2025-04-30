import pandas as pd

# Load the results file to filter based on additional tests
results_file_path = 'ссылка_на_файл'
df_results = pd.read_excel(results_file_path)

# Define the filtering criteria function based on new thresholds
def filter_model_results_v2(results):
    try:
        # if results["p-values"] >= 0.12:
        #     print(f"Model skipped due to p-values: {results['p-values']}")
        #     return False
        # if results["F-value"] <= 0.12:
        #     print(f"Model skipped due to F-value: {results['F-value']}")
        #     return False
        if results["Breusch-Pagan p-value"] <= 0.05:
            print(f"Model skipped due to Breusch-Pagan p-value: {results['Breusch-Pagan p-value']}")
            return False
        if results["White Test p-value"] <= 0.05:
            print(f"Model skipped due to White Test p-value: {results['White Test p-value']}")
            return False
        if results["Granger Causality p-value"] <= 0.05:
            print(f"Model skipped due to Granger Causality p-value: {results['Granger Causality p-value']}")
            return False
        if results["Normality Anderson-Darling p-value"] <= 0.05:
            print(f"Model skipped due to Anderson-Darling p-value: {results['Normality Anderson-Darling p-value']}")
            return False
        # if results["Normality KS p-value"] <= 0.05:
        #     print(f"Model skipped due to KS p-value: {results['Normality KS p-value']}")
        #     return False
        if not (0.25 <= results["Adjusted R-squared"] <= 0.95):
            print(f"Model skipped due to Adjusted R-squared: {results['Adjusted R-squared']}")
            return False
        return True
    except KeyError as e:
        print(f"Missing key in results: {e}")
        return False

# Apply the filtering criteria to the results
filtered_results = df_results[df_results.apply(filter_model_results_v2, axis=1)]

# Save the filtered results to a new Excel file
filtered_results_file_path = 'ссылка_на_файл'
filtered_results.to_excel(filtered_results_file_path, index=False)

filtered_results_file_path
