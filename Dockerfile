# FleetSense -- single image carrying the Flask app, the ML pipeline and the trained models.
#
# One image rather than separate app/model services: the ML is called as a library, not
# over a network. Splitting it would mean shipping every 113 MB recording between
# containers for no gain, since there is no independent scaling need. The clean separation
# that matters (app -> pipeline -> cost layer) is enforced by imports, not by process
# boundaries.

# ---------- build stage ----------
# Kept separate so the compilers stay out of the shipped image. numpy/scipy/scikit-learn
# normally install from wheels and need none of this, but if a wheel is ever missing for
# this Python version the build still succeeds instead of failing at the worst moment.
FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------- runtime stage ----------
FROM python:3.14-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /srv/fleetsense

# Created before the COPYs so ownership can be set per-copy. A `chown -R` afterwards would
# rewrite every file into a fresh layer, storing the 103 MB of models and the venv twice
# and adding ~104 MB to the image for nothing.
RUN useradd --create-home --shell /usr/sbin/nologin fleetsense \
    && mkdir -p /srv/fleetsense/app/instance \
    && chown fleetsense:fleetsense /srv/fleetsense /srv/fleetsense/app/instance

# Application code, the ML pipeline, and the trained models. Ordered so that the rarely
# changing model artifacts land in an earlier layer than the frequently edited app code.
COPY --chown=fleetsense:fleetsense src/mcc5/artifacts/ ./src/mcc5/artifacts/
COPY --chown=fleetsense:fleetsense src/mcc5/*.py ./src/mcc5/
COPY --chown=fleetsense:fleetsense src/maintenance/ ./src/maintenance/
COPY --chown=fleetsense:fleetsense app/ ./app/

USER fleetsense

EXPOSE 8000

# --timeout 180: a full recording is ~113 MB to receive plus ~5 s of analysis, which
#   comfortably exceeds gunicorn's 30 s default and would otherwise kill the worker
#   mid-request.
# --workers 2: each worker loads ~103 MB of models per request, so this trades throughput
#   the demo does not need for memory headroom it does.
# JSON form wrapping `sh -c exec`: sh is needed to expand ${PORT} (ECS and Elastic
# Beanstalk both inject it), and `exec` then REPLACES sh with gunicorn so that PID 1 is
# gunicorn itself. Without exec, SIGTERM on `docker stop` or an ECS task drain goes to sh,
# gunicorn never hears it, and the container is SIGKILLed after the grace period instead of
# finishing the request it is holding.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 180 --access-logfile - --error-logfile - --chdir app app:app"]
