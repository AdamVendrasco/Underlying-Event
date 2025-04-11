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

#COPY .globus /app/Underlying-Event/.globus
COPY Underlying-Event /app/Underlying-Event
RUN chmod -R a+rw /app/Underlying-Event
COPY Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt  /app/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt
COPY entry.sh /app/entry.sh

WORKDIR /app/Underlying-Event
RUN chmod +x /app/entry.sh

CMD ["/app/entry.sh"]
