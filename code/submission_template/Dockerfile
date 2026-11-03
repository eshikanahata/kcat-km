FROM python:3.12-slim-bookworm

# libgomp1: OpenMP runtime needed by LightGBM. Add other system libs here.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=app:app . /app
USER app

# The grader gives 2 vCPUs; no network; feature cache disabled at inference.
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 KCATKM_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 KCATKM_PRETRAINED=/app/pretrained \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf

CMD ["python", "/app/predict.py", "--help"]
