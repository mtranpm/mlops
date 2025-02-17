import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.autolog()

train_path = "loanapproval/data/train.csv"
test_path = "loanapproval/data/test.csv"

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

data = train_data.copy()
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col, encoder in label_encoders.items():
    data[col] = encoder.fit_transform(data[col])

# Split the data into features and target
X = data.drop(columns=['id', 'loan_status'])
y = data['loan_status']

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
X_train.head(), y_train.head()

params = {
    "solver": "liblinear",
    "max_iter": 2000,
    "multi_class": "auto",
    "random_state": 8888,
}

#train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred_val = lr.predict(X_val)

test_data_encoded = test_data.copy()
for col, encoder in label_encoders.items():
    test_data_encoded[col] = encoder.transform(test_data_encoded[col])

X_test = test_data_encoded.drop(columns=['id'])

y_pred=lr.predict(X_test)

#Metrics
y_val_pred=lr.predict_proba(X_val)[:,1]
roc_auc_val=roc_auc_score(y_val, y_val_pred)

accuracy_val=accuracy_score(y_val,y_pred_val)

# Predict probabilities on test data
test_pred_proba = lr.predict_proba(X_test)[:, 1]

# Threshold at 0.5 to convert probabilities to binary loan_status predictions
test_pred = (test_pred_proba >= 0.5).astype(int)
test_data_classification=pd.DataFrame({
    'id': test_data['id'],
    'loan_status': test_pred
})
test_data_classification.to_csv('output.csv',index=False)

confusion_metrics=confusion_matrix(y_val,y_pred_val)


#metrics={"accuracy on val":accuracy_val, "roc auc val": roc_auc_val.item(), "CF":confusion_metrics.item()}

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
#mlflow.set_experiment(experiment)

experiment_name= "loanapp_lr"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
  experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id,run_name='ver1'):
    mlflow.log_params(params)
    mlflow.log_metric("accuracy",accuracy_val)
    mlflow.log_metric("roc auc score",roc_auc_val)
    
    plt.figure()
    sns.heatmap(confusion_metrics, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    cm_image_path = "confusion_matrix.png"
    plt.savefig(cm_image_path)
    mlflow.log_artifact(cm_image_path)

    mlflow.set_tag("release.version","1.0")
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info= mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="loanapproval",
        signature=signature,
        input_example=X_train,
        registered_model_name="loanapp-v1",
    )
