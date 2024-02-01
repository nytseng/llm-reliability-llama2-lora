## Calculates f1 score of PE, KE, LCE (Binary)
# code from FLAN_T5_Training_v3.ipynb, author: Minghao Zhou
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, classification_report
             
results_df = pd.read_csv('predict_output_baseline.csv')
results_df['PE_Actual'] = results_df['PE_Actual'].str.lower()
results_df['PE_Predicted'] = results_df['PE_Predicted'].str.lower()
results_df['KE_Actual'] = results_df['KE_Actual'].str.lower()
results_df['KE_Predicted'] = results_df['KE_Predicted'].str.lower()
results_df['LCE_Actual'] = results_df['LCE_Actual'].str.lower()
results_df['LCE_Predicted'] = results_df['LCE_Predicted'].str.lower()

# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    # Precision, Recall for each class
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=["acceptable", "unacceptable"], zero_division=1)
    # Classification report for detailed metrics
    class_report = classification_report(y_true, y_pred, labels=["acceptable", "unacceptable"], zero_division=1)

    return accuracy, f1, precision, recall, class_report

accuracy_pe, f1_pe, precision_pe, recall_pe, report_pe = calculate_metrics(results_df['PE_Actual'], results_df['PE_Predicted'])
accuracy_ke, f1_ke, precision_ke, recall_ke, report_ke = calculate_metrics(results_df['KE_Actual'], results_df['KE_Predicted'])
accuracy_lce, f1_lce, precision_lce, recall_lce, report_lce = calculate_metrics(results_df['LCE_Actual'], results_df['LCE_Predicted'])

# Print the results
print("Metrics for PE:")
print(f"Accuracy: {accuracy_pe}, F1 Score: {f1_pe}")
print(f"Precision: {precision_pe}, Recall: {recall_pe}")
print(f"Classification Report:\n{report_pe}")

print("\nMetrics for KE:")
print(f"Accuracy: {accuracy_ke}, F1 Score: {f1_ke}")
print(f"Precision: {precision_ke}, Recall: {recall_ke}")
print(f"Classification Report:\n{report_ke}")

print("\nMetrics for LCE:")
print(f"Accuracy: {accuracy_lce}, F1 Score: {f1_lce}")
print(f"Precision: {precision_lce}, Recall: {recall_lce}")
print(f"Classification Report:\n{report_lce}")