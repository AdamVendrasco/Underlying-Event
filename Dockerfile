FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
COPY Underlying-Event /app/Underlying-Event
COPY entry.sh /app/entry.sh

RUN chmod +x /app/entry.sh
RUN conda env create -f environment.yml
RUN mkdir /app/Underlying-Event/root_files/
RUN mkdir /app/Underlying-Event/plots/

CMD ["/app/entry.sh"]

