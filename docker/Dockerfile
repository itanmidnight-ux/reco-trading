# syntax=docker/dockerfile:1.7
FROM python:3.11.11-slim-trixie AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 && \
    pip install --requirement /tmp/requirements.txt

COPY . /app

CMD ["python", "trading_system/app/main.py"]
