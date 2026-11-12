FROM sport-base:v1
WORKDIR /participant
COPY participant/ /participant/
RUN chmod +x /participant/run_all.sh \
    && PYTHONPATH=/participant/pingpang /opt/conda/envs/conda_env/bin/python3 /participant/pingpang/integrity.py --verify \
    && PYTHONPATH=/participant/basketball /opt/conda/envs/conda_env/bin/python3 /participant/basketball/integrity.py --verify
ENTRYPOINT ["/participant/run_all.sh"]
