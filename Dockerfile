# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Copy project files ----------
COPY . /app

# ---------- Install dependencies ----------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------- Expose Streamlit Port ----------
EXPOSE 8501

# ---------- Run the Streamlit App ----------
CMD ["streamlit", "run", "app/app.py"]
