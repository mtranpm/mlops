import mlflow
logged_model = 'runs:/d29098d4e52a4ff3ac83d8cf2e3f12a6/loanapproval'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data=pd.read_csv('out.csv')
print(loaded_model.predict(pd.DataFrame(data)))