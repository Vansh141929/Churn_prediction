# Deploy the churn app with Streamlit Community Cloud

The repository is ready to deploy: the app entry point is `app.py`, dependencies are listed in `requirements.txt`, and the trained model and scaler are stored under `models/`.

## Publish it

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account if prompted.
2. Select **Create app** and choose the `Vansh141929/Churn_prediction` repository.
3. Set the branch to `main` and the main file path to `app.py`.
4. Select **Deploy**. Streamlit Cloud will install `requirements.txt` and start the app.

The app does not require secrets or additional configuration. The included `.streamlit/config.toml` sets the dashboard's dark theme.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

If deployment fails while loading a saved model, check that the installed `scikit-learn` and `xgboost` versions are compatible with the versions used when `models/churn_model.pkl` was created.
