# AI-Powered Network Anomaly & Intrusion Detection System

A production-oriented End of Studies Project (PFE) that detects suspicious
network behavior using Zeek flow logs, Scikit-Learn, Elasticsearch, Kibana, and
Docker.

The system trains a Random Forest classifier, tails Zeek-style JSON `conn.log`
records in real time, extracts flow features, performs inference, and sends both
analyzed flows and high-risk alerts to Elasticsearch for Kibana visualization.

## Architecture Overview

```text
Network Traffic
      |
      v
Zeek Sensor
      |
      v
JSON conn.log
      |
      v
Python ML Engine
      |
      +--> Feature Extraction
      +--> Random Forest Inference
      +--> Alert Scoring
      |
      v
Elasticsearch
      |
      v
Kibana Dashboards
```

## Components

| Component | Technology | Purpose |
|---|---|---|
| Network Sensor | Zeek | Produces structured JSON flow logs |
| ML Training | Python, Scikit-Learn | Trains a Random Forest IDS classifier |
| Real-Time Detector | Python | Tails `conn.log`, extracts features, runs inference |
| Storage | Elasticsearch | Stores analyzed flows and security alerts |
| Visualization | Kibana | Enables dashboards, filtering, and SOC-style review |
| Deployment | Docker Compose | Provides reproducible local deployment |

## One-Command Setup

From the project root:

```bash
docker compose up --build
```

This starts:

- Elasticsearch on `http://localhost:9200`
- Kibana on `http://localhost:5601`
- ML engine container
- Mock Zeek log producer
- Real-time anomaly detector

The ML engine automatically:

1. Installs Python dependencies.
2. Uses the included mock CIC-IDS2017-like seed dataset.
3. Trains a Random Forest classifier.
4. Saves the model to `ml_engine/models/random_forest_ids.pkl`.
5. Tails `zeek_logs/conn.log`.
6. Sends flows and alerts to Elasticsearch.

The repository includes starter data so the project is not empty:

- `ml_engine/data/mock_cic_ids2017.csv`: labeled demo training flows.
- `zeek_logs/conn.log`: sample Zeek-style JSON flow events.

During a full demo, the detector can also keep appending fresh mock Zeek events
when `MOCK_LOG_PRODUCER=true`.

## Elasticsearch Indices

The detector creates two indices:

| Index | Description |
|---|---|
| `ids-flows` | All analyzed network flows with prediction scores |
| `ids-alerts` | High-risk anomalous flows classified as attacks |

## Kibana Demo

Open Kibana:

```text
http://localhost:5601
```

Create data views:

```text
ids-flows
ids-alerts
```

Use `@timestamp` as the time field.

Suggested visualizations:

- Alert count over time
- Top source IPs by alert volume
- Top destination ports involved in anomalies
- Average anomaly score over time
- Severity distribution
- Flow classification ratio: benign vs attack

## Running Training Manually

```bash
cd ml_engine
pip install -r requirements.txt
python train_model.py
```

Outputs:

```text
ml_engine/models/random_forest_ids.pkl
ml_engine/models/training_metrics.json
```

## Running Detection Manually

Start Elasticsearch first, then run:

```bash
cd ml_engine
python real_time_detector.py
```

Useful environment variables:

| Variable | Default | Description |
|---|---|---|
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `MODEL_PATH` | `models/random_forest_ids.pkl` | Trained model path |
| `ZEEK_CONN_LOG` | `../zeek_logs/conn.log` | Zeek JSON log path |
| `ALERT_INDEX` | `ids-alerts` | Elasticsearch alert index |
| `FLOW_INDEX` | `ids-flows` | Elasticsearch flow index |
| `MOCK_LOG_PRODUCER` | `true` | Enables synthetic Zeek logs |
| `ALERT_THRESHOLD` | `0.70` | Attack probability threshold |

## Zeek Log Format

The detector expects JSON lines similar to:

```json
{
  "ts": "2026-06-10T21:30:00Z",
  "uid": "C123456789",
  "id.orig_h": "192.168.1.20",
  "id.orig_p": 51544,
  "id.resp_h": "10.0.0.5",
  "id.resp_p": 443,
  "proto": "tcp",
  "service": "ssl",
  "duration": 1.23,
  "orig_bytes": 3200,
  "resp_bytes": 18400,
  "orig_pkts": 22,
  "resp_pkts": 31,
  "missed_bytes": 0,
  "history": "ShADadFf"
}
```

## Feature Engineering

The model uses flow-level metadata only:

- Duration
- Originator bytes
- Responder bytes
- Originator packets
- Responder packets
- Missed bytes
- Zeek history length
- Protocol flags
- Service flags

This design works with encrypted traffic because it does not require payload
inspection.

## Security Notes

This project is suitable for PFE demonstration and controlled lab environments.

For production deployment, improve the following:

- Use real CIC-IDS2017, UNSW-NB15, or organization-specific flow data.
- Add model drift monitoring.
- Add authentication and TLS for Elasticsearch and Kibana.
- Replace the mock log producer with a real Zeek sensor.
- Add CI/CD, unit tests, and model validation gates.
- Add alert deduplication and case management integration.
- Tune thresholds using precision, recall, F1-score, and false positive rate.

## Project Value

This project demonstrates:

- Cybersecurity monitoring architecture
- Flow-based intrusion detection
- Machine learning model training
- Real-time inference
- Elasticsearch/Kibana observability
- Dockerized MLOps deployment
- Professional security engineering documentation
