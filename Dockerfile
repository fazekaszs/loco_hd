FROM mambaorg/micromamba:latest AS builder

USER root
WORKDIR /app

# Move files into the image.
COPY src ./src
COPY loco_hd ./loco_hd
COPY primitive_typings ./primitive_typings
COPY Cargo.toml pyproject.toml ./

# Install dependencies using mamba.
RUN micromamba install -y -c conda-forge -n base python=3.12 rust maturin

RUN micromamba run -n base maturin build --release -o .

FROM python:3.12-slim as deployer

WORKDIR /app

COPY --from=builder /app/loco_hd-*.whl .

RUN pip install loco_hd-*.whl

ENTRYPOINT ["python", "-m", "loco_hd"]