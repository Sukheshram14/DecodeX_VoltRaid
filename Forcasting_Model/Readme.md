
## 1. Python Virtual Environment

### Create Virtual Environment

```bash
python -m venv myenv
```

OR

```bash
py -m venv myenv
```

### Activate Environment (Windows)

```bash
.\myenv\Scripts\activate
```

### Deactivate Environment

```bash
deactivate
```


## 2. Required Libraries

Add all dependencies to `requirements.txt`:

```
pandas
numpy
scikit-learn
prophet
plotly
streamlit
joblib

```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### List Installed Packages

```bash
python -m pip list
```

---

## 3. Running the Streamlit Chatbot

### Start the Application

```bash
streamlit run app.py
```

