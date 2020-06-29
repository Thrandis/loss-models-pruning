FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

ADD requirements.txt /tmp/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    conda clean -ya
