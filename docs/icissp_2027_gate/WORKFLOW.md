# Ausführbarer technischer Workflow

Alle folgenden Aufrufe erfolgen im DOS-Root nach dem Einbau. Resultate unter `new/runs/cache_isolation_gate/` ablegen, Serverlogs werden automatisch unter `new/logs/cache_isolation_gate/server/` geschrieben. Kein Aufruf dieses Pakets startet eine parallele Zusatzlastkampagne.

## 1. Offline-Abnahme

`EINBAU.md` beschreibt den Preflight. Die reinen Auswertewerkzeuge brauchen nur die Python-Standardbibliothek; kein pip install. Der Server braucht die bereits vorhandene vLLM-Umgebung. Die Cache-Probe braucht deren vorhandenes httpx und den originalen Prefill-Executor.

Die Vorlagen unter `new/scripts/cache_isolation_gate/contracts/` in den Runordner kopieren und dort ausfüllen. Die Paketvorlagen nicht als ausgefüllte Verträge präsentieren.

```bash
python3 -B new/scripts/cache_isolation_gate/validate_contract.py --contract new/runs/cache_isolation_gate/study_contract.json
```

Mit offenen Nullfeldern scheitert die Prüfung absichtlich. Die erfolgreiche Vollständigkeitsprüfung belegt noch keine zeitliche Vorabfestlegung und keinen Runtime-Gate-Pass.

## 2. Server und reine Beobachtung

Server wie in EINBAU.md starten. In einem zweiten Terminal nach bereitgemeldetem Server:

```bash
mkdir -p new/runs/cache_isolation_gate
```

```bash
python3 -B new/scripts/cache_isolation_gate/record_metrics.py --output new/runs/cache_isolation_gate/metrics_function_check.jsonl --duration-s 60
```

Der Recorder liest ausschließlich `http://127.0.0.1:8000/metrics`, standardmäßig alle 0,25 Sekunden. Neue Dateien werden exklusiv angelegt, vorhandene nicht überschrieben. Bei HTTP-Fehlern bleiben Fehlerzeilen erhalten. Bei normalem Ende/SIGINT/SIGTERM wird ein Abschlussdatensatz geschrieben; SIGKILL kann keine saubere Abschlusszeile garantieren. Maximaldauer 3600 Sekunden, Bodylimit 2 MiB. Keine Completion-Requests.

## 3. Kleine Cache-Funktionsprobe

Eine JSON-Datei `public_prompt_token_ids.json` mit einem vorhandenen öffentlichen Testpräfix als Token-ID-Liste bereitstellen. 1 bis 128 Tokens erlaubt; zu kurze Präfixe unterhalb vollständiger KV-Blöcke sind für einen Hitnachweis ungeeignet. IDs müssen mit dem tatsächlich geladenen Modell übereinstimmen. Keine privaten Prompts verwenden.

Mit identischem Serverschlüssel in VLLM_API_KEY und der vorhandenen Python-Umgebung:

```bash
/home/rock/projects/porto/.venv-py313-repair/bin/python3 -B new/scripts/cache_isolation_gate/cache_smoke.py --root /home/rock/projects/DOS --prompt-token-ids new/runs/cache_isolation_gate/public_prompt_token_ids.json --output new/runs/cache_isolation_gate/cache_function_check.json
```

Die Probe sendet maximal drei strikt sequenzielle Requests mit je zwei Ausgabetokens: Salt A, Salt A, Salt B. Sie nutzt den unveränderten historischen Raw-SSE-Executor. Nach jedem Request werden Metriken gelesen. Das belegt weder tenantweise Ressourcenquoten noch Cross-Request-Interferenz. Vorbestehender Cachezustand, vollständige Blockgrenzen und verzögertes Metriklogging können einen Befund unentscheidbar machen. Deshalb steht `cache_rule_validated` immer false; tatsächliche Gegenstände anhand der Evidenz beurteilen. Exit-Code 0 bedeutet vollständige Requestausführung, nicht bestätigte Cache-Regel.

## 4. Metriken und Iterationen auswerten

```bash
python3 -B new/scripts/cache_isolation_gate/summarize_metrics.py --input new/runs/cache_isolation_gate/metrics_function_check.jsonl > new/runs/cache_isolation_gate/metrics_summary.json
```

```bash
python3 -B new/scripts/cache_isolation_gate/parse_iteration_logs.py --log PFAD_ZUM_SERVERLOG > new/runs/cache_isolation_gate/iterations.json
```

Zwei tatsächlich vorliegende vergleichbare Logdateien:

```bash
python3 -B new/scripts/cache_isolation_gate/compare_budget_logs.py --small-log PFAD_KLEINES_BUDGET --large-log PFAD_GROSSES_BUDGET
```

Der Vergleich zählt geplante Context-Tokens pro Iteration. Ein Unterschied kann auch durch zufällige Ankunftszeiten oder Cachezustände entstehen. Wirksamkeit deshalb nur bei kontrollierter angebotener Arbeit, gleicher effektiver Kapazität und reproduzierbarem Schedulingunterschied abnehmen. Die Cache-Probe allein ist kein ausreichender Budgetnachweis.

## 5. Bereits vorhandene neue Studienepisoden auswerten

Ausgefüllten Studienvertrag und eine ausgefüllte Episode-Map bereitstellen. Jeder Eintrag referenziert eine bestehende Episode und tatsächliche Konfigurationsevidenz. Pfade relativ zum Ort der Map; absolute Pfade sind ebenfalls möglich.

```bash
python3 -B new/scripts/cache_isolation_gate/build_endpoint_table.py --contract new/runs/cache_isolation_gate/study_contract.json --episode-map new/runs/cache_isolation_gate/episode_map.json > new/runs/cache_isolation_gate/endpoints.json
```

```bash
python3 -B new/scripts/cache_isolation_gate/analyze_pairs.py --input new/runs/cache_isolation_gate/endpoints.json > new/runs/cache_isolation_gate/paired_analysis.json
```

Der erste Schritt führt keine Requests aus. Der zweite verlangt vollständige achtzellige Replikatblöcke mit je Referenz-/Zusatzlastbeobachtung. Ausfälle werden nicht still ausgeschlossen. Zu wenige oder unvollständige Daten erzeugen einen Fehler. Verträge und Metadaten werden nicht automatisch zu kausalen Laufzeitnachweisen.

Die Erstdiagnose besteht somit aus Quellen-/Umgebungsprüfung, normalem Serverstart, reiner Beobachtung und begrenzter Cache-Funktionsprobe. Die eigentliche Acht-Zellen-Messung beginnt erst nach separat bestätigtem Gate; sie wird von diesem Paket nicht automatisch gestartet.
