FROM python:3.12-slim

ARG PROCESSOR_MODEL_ID
ARG IE_MODEL_ID
ARG S3_URI="s3://kaneel-sagemaker-testing/ie-model-artifacts/$IE_MODEL_ID/output/model.tar.gz"
ARG AWS_REGION
ARG HF_TOKEN

ENV PROCESSOR_MODEL_ID=$PROCESSOR_MODEL_ID

ENV IE_MODEL_ID=$IE_MODEL_ID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.models \
    && if [ -n "$HF_TOKEN" ]; then \
         hf download "$PROCESSOR_MODEL_ID" --token "$HF_TOKEN" --local-dir "/app/.models/$PROCESSOR_MODEL_ID"; \
       else \
         hf download "$PROCESSOR_MODEL_ID" --local-dir "/app/.models/$PROCESSOR_MODEL_ID"; \
       fi

RUN apt-get update && apt-get install -y awscli tar && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=aws_access_key_id \
    --mount=type=secret,id=aws_secret_access_key \
    --mount=type=secret,id=aws_session_token \
    mkdir -p /app/.models \
    && AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_access_key_id) \
    AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret_access_key) \
    AWS_SESSION_TOKEN=$(cat /run/secrets/aws_session_token) \
    aws s3 cp $S3_URI /app/.models/$IE_MODEL_ID/model.tar.gz \
    && tar -xzf /app/.models/$IE_MODEL_ID/model.tar.gz -C /app/.models/$IE_MODEL_ID \
    && rm /app/.models/$IE_MODEL_ID/model.tar.gz

# Install dependencies first for better layer caching
COPY src/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy source
COPY src /app/src

# Run from /app/src so config.get_models_dir() -> "../.models" resolves to /app/.models
WORKDIR /app/src

# Create writable runtime dirs used by the script
RUN mkdir -p /app/.cache /app/tmp \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "main.py"]
