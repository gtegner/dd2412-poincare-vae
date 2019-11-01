FROM pytorch/pytorch

WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD . /app
ARG EXPERIMENT

ENTRYPOINT ["python", "experiments.py"]

