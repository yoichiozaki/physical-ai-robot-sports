# syntax=docker/dockerfile:1
# Shared Dev Container for physical-ai-robot-sports (air-hockey / robot-arm-tennis)
# - Python 3.11 + MuJoCo (headless EGL/OSMesa rendering on CPU via Mesa llvmpipe)
# - CPU build of PyTorch (SB3 MlpPolicy PPO is faster on CPU); GPU is optional at runtime
# - No credentials are baked into this image.
FROM python:3.11.17-slim-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MPLBACKEND=Agg \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# System libraries:
#  - libegl/libgl/mesa: headless OpenGL for MuJoCo offscreen rendering (EGL, fallback OSMesa)
#  - ffmpeg: video encoding, fonts-ipaexfont: Japanese text in video/figures/PDF (embedded TTF)
#  - poppler-utils: PDF page rasterisation for report validation
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion ca-certificates curl git make sudo less procps \
        ffmpeg \
        libegl1 libgl1 libgles2 libglvnd0 libegl-mesa0 libgl1-mesa-dri libosmesa6 \
        fonts-ipaexfont-gothic fonts-ipaexfont-mincho \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && python -m venv ${VIRTUAL_ENV} \
    && chown -R ${USERNAME}:${USERNAME} ${VIRTUAL_ENV} \
    && printf 'export VIRTUAL_ENV=%s\nexport PATH=%s/bin:$PATH\nexport MUJOCO_GL=${MUJOCO_GL:-egl}\nexport PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}\nexport MPLBACKEND=Agg\n' \
        "${VIRTUAL_ENV}" "${VIRTUAL_ENV}" > /etc/profile.d/10-venv.sh

USER ${USERNAME}

# The workspace is bind-mounted from the host (different owner): let git (run_info.json
# records the commit) operate on it.
RUN git config --global --add safe.directory /workspaces/physical-ai-robot-sports
# Pinned dependencies (full lock incl. transitive packages, CPU torch wheel index).
COPY --chown=${USERNAME}:${USERNAME} requirements/ /tmp/requirements/
RUN pip install --upgrade pip==26.2.1 \
    && pip install -r /tmp/requirements/lock.txt

# Official Google Colab CLI (needs Python >= 3.12): isolated by uv with a managed Python,
# so the project itself stays on Python 3.11. Credentials are NOT part of the image
# (OAuth token lives in ~/.config/colab-cli, mounted as a volume by devcontainer.json).
ENV PATH=/home/${USERNAME}/.local/bin:$PATH
RUN pip install uv==0.12.24 \
    && uv tool install --python 3.12 google-colab-cli==0.7.4 \
    && mkdir -p /home/${USERNAME}/.config/colab-cli

WORKDIR /workspaces/physical-ai-robot-sports
