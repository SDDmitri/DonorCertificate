FROM python:3.12
RUN python -m pip install --upgrade --root-user-action ignore pip
RUN --mount=type=bind,source=torch_req.txt,target=/tmp/torch_req.txt \
    pip install -r /tmp/torch_req.txt --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install -r /tmp/requirements.txt
RUN mkdir /rot_imgs

