FROM waggle/plugin-base:1.1.0-ml-cuda11.0-amd64

# install pip dependencies:
COPY requirements.txt /app/
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy plugin app content:
COPY attention_cae.py  calibration_data.py  main.py  \
     online_models.py  samplers.py  test.py \ 
     /app/
COPY test /app/test/
COPY calibration_data /app/calibration_data/
COPY saved_weights /app/saved_weights/

# Set SAGE environment variables:
ARG SAGE_STORE_URL="HOST"
ARG SAGE_USER_TOKEN="-10"
ARG BUCKET_ID_MODEL="BUCKET_ID_MODEL"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    SAGE_USER_TOKEN=${SAGE_USER_TOKEN} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

# Establish entrypoint:
# ----- Arguments -----
# (TODO) 
WORKDIR /app
ENTRYPOINT ["python3", "/app/main.py"]
