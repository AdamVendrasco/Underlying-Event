FROM rootproject/root:6.30.06-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    xrootd-client \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt \
    && pip install XRootD

COPY .globus /app/Underlying-Event/.globus
COPY Underlying-Event /app/Underlying-Event
COPY entry.sh /app/entry.sh

WORKDIR /app
RUN mkdir -p /app/Underlying-Event/root_files/ /app/Underlying-Event/plots/
RUN chmod +x /app/entry.sh

CMD ["/app/entry.sh"]
