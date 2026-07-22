# SDPO_ReAct code_interpreter sandbox -- one long-lived container per training
# job, exposing exactly one port (see ../run_sandbox.sh). Preinstalls the math
# stack (sympy/numpy/scipy) so AIME-style rollouts don't pay import latency or
# need network access on every tool call.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sandbox_server.py .

# Unprivileged user: code submitted by the (untrusted, still-training) policy
# runs as this user inside the container, not root.
RUN useradd -m -u 1000 sandbox
USER sandbox

EXPOSE 8420
CMD ["uvicorn", "sandbox_server:app", "--host", "0.0.0.0", "--port", "8420"]
