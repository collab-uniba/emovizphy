FROM python:3.8

WORKDIR /emovizphy
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./ ./

EXPOSE 20000

ENTRYPOINT ["python", "visualization_tool.py"]