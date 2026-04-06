# Loan Default Prediction — MLOps Project

A production-ready MLOps pipeline for predicting loan defaults. The project trains and compares three models (Logistic Regression, Decision Tree, LightGBM) on 10,000 customer records, tracks all experiments with MLflow, selects the best model by AUC-ROC, and serves predictions through a Streamlit web app. Infrastructure is containerised with Docker and deployed to Azure Container Apps via a GitHub Actions CI/CD pipeline.

## File Structure

```
loan-default-mlops/
├── data/
│   └── Loan_Data.csv          ← Raw dataset (10k rows)
├── notebook.ipynb              ← ALL ML work: EDA, preprocessing, training, evaluation
├── app.py                      ← Streamlit prediction app
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci_cd.yml           ← Lint → Build & Push → Deploy
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/loan-default-mlops.git
cd loan-default-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook (trains models + saves artefacts to models/)
jupyter notebook notebook.ipynb

# 4. Launch the Streamlit app
streamlit run app.py
```

## Docker Compose

```bash
# Build and start both services (Streamlit app + MLflow UI)
docker-compose up --build

# Streamlit app  → http://localhost:8501
# MLflow UI      → http://localhost:5000
```

> **Note:** Run the notebook first to generate the `models/` directory and trained artefacts before building the Docker image.

## GitHub Actions — Required Secrets

Set the following secrets in your repository (**Settings → Secrets and variables → Actions**):

| Secret                     | Description                                  |
|----------------------------|----------------------------------------------|
| `AZURE_REGISTRY_URL`       | ACR login server, e.g. `myacr.azurecr.io`   |
| `AZURE_REGISTRY_USERNAME`  | ACR admin username                           |
| `AZURE_REGISTRY_PASSWORD`  | ACR admin password                           |
| `AZURE_CREDENTIALS`        | JSON output of `az ad sp create-for-rbac`    |

## Azure Setup

```bash
# Create a resource group
az group create --name mlops-rg --location westeurope

# Create an Azure Container Registry
az acr create --resource-group mlops-rg --name myacr --sku Basic --admin-enabled true

# Create an Azure Container App environment
az containerapp env create --name mlops-env --resource-group mlops-rg --location westeurope

# Create the Container App (first deploy)
az containerapp create \
  --name loan-default-app \
  --resource-group mlops-rg \
  --environment mlops-env \
  --image myacr.azurecr.io/loan-default-app:latest \
  --target-port 8501 \
  --ingress external \
  --registry-server myacr.azurecr.io \
  --registry-username <ACR_USERNAME> \
  --registry-password <ACR_PASSWORD>

# Generate service principal credentials for CI/CD
az ad sp create-for-rbac --name "github-actions-sp" \
  --role contributor \
  --scopes /subscriptions/<SUB_ID>/resourceGroups/mlops-rg \
  --sdk-auth
```

Copy the JSON output of the last command into the `AZURE_CREDENTIALS` secret.
 
