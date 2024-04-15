import pandas as pd
import mlflow
import mlflow.sklearn
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


low=pd.read_csv('../../../data/processed/to_ML_models/LowRecords.csv',index_col=0)
med=pd.read_csv('../../../data/processed/to_ML_models/MediumRecords.csv',index_col=0)
high=pd.read_csv('../../../data/processed/to_ML_models/HighRecords.csv',index_col=0)
low['Degree_of_Plagiarism']= 0 # Low
med['Degree_of_Plagiarism']=1 # Medium 
high['Degree_of_Plagiarism']=2 # High
Final_data=pd.concat([low,med,high])
def bias_variance_error(model,X_train,y_train,X_test, y_test):
    X_train,y_train,X_test,y_test=X_train.values,y_train.values,X_test.values,y_test.values
    avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(
    model, X_train, y_train, X_test, y_test, 
    loss='0-1_loss',
    random_seed=123)
    return [avg_bias,avg_variance]
X, y =  Final_data[['Hyperbole', 'Text_similarity', 'Readability',
       'Style_Consistency']], Final_data[['Degree_of_Plagiarism']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.set_experiment("Plagiarism_Detection_Evaluation")
labels={"0":"Low","1":"Medium","2":"High"}

with mlflow.start_run(run_name="LogisticRegression"):
    lr_param_grid = {
    'C': [0.1, 1, 10],  
    'max_iter': [100, 500, 1000]
                    }
    for i in lr_param_grid['C']:
        for j in lr_param_grid['max_iter']:
            lr_model = LogisticRegression(C=i,max_iter=j)
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            report = classification_report(y_test, y_pred_lr, output_dict=True)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if label in labels.keys():
                            mlflow.log_metric(f"{labels[label]}_{metric_name}", metric_value)


            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_lr))
            mlflow.log_metric("precision", precision_score(y_test, y_pred_lr, average="weighted"))
            mlflow.log_metric("recall", recall_score(y_test, y_pred_lr, average="weighted"))
            mlflow.log_metric("f1", f1_score(y_test, y_pred_lr, average="weighted"))
            mlflow.log_metric("C", i)
            mlflow.log_metric("Max_iterations", j)
            mlflow.log_metric("Bias_Error",bias_variance_error(lr_model,X_train,y_train,X_test,y_test)[0])
            mlflow.log_metric("Variance_Error",bias_variance_error(lr_model,X_train,y_train,X_test,y_test)[1])


with mlflow.start_run(run_name="RandomForest"):
    rf_param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    for i in rf_param_grid['n_estimators']:
        for j in rf_param_grid['max_depth']:
            for k in rf_param_grid['min_samples_split']:
                rf_model = RandomForestClassifier(n_estimators=i, max_depth=j, min_samples_split=k)
                rf_model.fit(X_train, y_train)
                y_pred_rf = rf_model.predict(X_test)
                report = classification_report(y_test, y_pred_rf, output_dict=True)
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            if label in labels.keys():
                                mlflow.log_metric(f"{labels[label]}_{metric_name}", metric_value)
            
                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_rf))
                mlflow.log_metric("precision", precision_score(y_test, y_pred_rf, average="weighted"))
                mlflow.log_metric("recall", recall_score(y_test, y_pred_rf, average="weighted"))
                mlflow.log_metric("f1", f1_score(y_test, y_pred_rf, average="weighted"))
                mlflow.log_metric("N_estimators", i)
                mlflow.log_metric("Max_depth", j)
                mlflow.log_metric("Min_samples_split", k)
                mlflow.log_metric("Bias_Error",bias_variance_error(rf_model,X_train,y_train,X_test,y_test)[0])
                mlflow.log_metric("Variance_Error",bias_variance_error(rf_model,X_train,y_train,X_test,y_test)[1])


with mlflow.start_run(run_name="SupportVectorMachines"):
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    }
    for i in svm_param_grid['C']:
        for j in svm_param_grid['gamma']:
            for k in svm_param_grid['kernel']:
                svm_model = svm.SVC(C=i, gamma=j, kernel=k)
                svm_model.fit(X_train, y_train)
                y_pred_svm = svm_model.predict(X_test)
                report = classification_report(y_test, y_pred_svm, output_dict=True)
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            if label in labels.keys():
                                mlflow.log_metric(f"{labels[label]}_{metric_name}", metric_value)
           

                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_svm))
                mlflow.log_metric("precision", precision_score(y_test, y_pred_svm, average="weighted"))
                mlflow.log_metric("recall", recall_score(y_test, y_pred_svm, average="weighted"))
                mlflow.log_metric("f1", f1_score(y_test, y_pred_svm, average="weighted"))
                mlflow.log_metric("C", i)
                mlflow.log_metric("Gamma", j)
                mlflow.log_metric("Bias_Error",bias_variance_error(svm_model,X_train,y_train,X_test,y_test)[0])
                mlflow.log_metric("Variance_Error",bias_variance_error(svm_model,X_train,y_train,X_test,y_test)[1])


with mlflow.start_run(run_name="NaiveBayes"):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    report = classification_report(y_test, y_pred_nb, output_dict=True)
    for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            if label in labels.keys():
                                mlflow.log_metric(f"{labels[label]}_{metric_name}", metric_value)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_nb))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_nb, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_nb, average="weighted"))
    mlflow.log_metric("f1", f1_score(y_test, y_pred_nb, average="weighted"))
    mlflow.log_metric("Bias_Error",bias_variance_error(nb_model,X_train,y_train,X_test,y_test)[0])
    mlflow.log_metric("Variance_Error",bias_variance_error(nb_model,X_train,y_train,X_test,y_test)[1])



mlflow.end_run()

print("Models trained and metrics logged successfully!")
