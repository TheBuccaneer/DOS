# Prefill Shape v1

Eigenständiger Runner für die Frage, ob die zeitliche Konzentration identischer zusätzlicher Prefill-Last decodierende Requests bei Budgets 512 und 2048 unterschiedlich verzögert. Der vorhandene Raw-SSE-Executor wird unverändert und per SHA-256 geprüft geladen. Historische Runner, Messdaten und Verträge werden nicht geändert.

## Festgelegtes Design

- vLLM 0.17.1, Llama-3.1-8B-Instruct, BF16, eine RTX 3090, CPU-Offload 0, Prefix-Caching **aus**.
- Vier Victims, jeweils 256 Input- und 64 Output-Tokens, ausgehend von einem gemeinsamen Startzeitpunkt `t0` gleichzeitig als Tasks gestartet.
- Burst und Verteilt enthalten dieselben vier zusätzlichen Requests mit jeweils 2.048 Input- und 16 Output-Tokens. Kontrolle enthält nur die Victims.
- Diagnostik: Timer-Deadline `t0 + 800 ms`. Burst startet alle vier zusätzlichen Tasks an dieser Deadline. Verteilt startet sie an den absoluten Deadlines `t0 + {800,900,1000,1100} ms`. Es wird nicht auf vorherige zusätzliche Requests gewartet.
- Diagnosepläne enthalten genau einen Dreierblock für das ausgewählte Budget 512 oder 2048 (`prepare --mode diagnostic --budget ...`); der Standard bleibt 2048. Beide Budgets verwenden dieselbe Last und dieselben technischen und wissenschaftlichen Prüfregeln.
- Der Timer liest keine Victim-Fortschritte. Auch bei bereits beendeten Victims werden die geplanten zusätzlichen Requests versendet. Fehlende Decoding-Exposition wird anschließend ausgewiesen.
- Prompts und Request-Seeds sind innerhalb jedes Dreierblocks identisch. Auch gleich indizierte Blöcke beider Budgets verwenden gepaarte Prompts/Seeds. Zwischen Blockindizes werden neue Seeds und Prompts abgeleitet. Die Bedingungsreihenfolge wird blockweise randomisiert.
- Für eine Hauptkampagne kann explizit ein Plan mit zehn Dreierblöcken je Budget erzeugt werden: 60 Episoden, 400 Completion-Requests. Timer und Verteilt-Abstand müssen dafür separat vorab angegeben werden. `run-block` führt immer nur **einen** Dreierblock aus; ein Hauptblock verlangt zusätzlich `--allow-main-block`.

Zur Kontrolle einer möglichen KV-Kapazitätskonfundierung werden bei beiden Budgets dieselben 1.024 KV-Blöcke à 16 Tokens verwendet (16.384 Tokens). Die acht vollständig belegten Sequenzen benötigen nominal höchstens 9.536 Tokens. Max. Sequenzlänge 4.096, max. Sequenzzahl 16, Tensor Parallelism 1 sowie Partial-Prefill-Einstellungen bleiben konstant. Das ist eine vorab gewählte Ressourcenbedingung, kein Ersatz für die tatsächliche Prüfung auf Preemptions.

## Primärmetrik

```text
T64_cohort_ms = (max(receipt_ns(V0,64), ..., receipt_ns(V3,64)) - t0) / 1e6
```

Dies ist die Zeit bis zum Token 64 des langsamsten Victims relativ zum gemeinsamen Victim-Start. Individuelle Token-64-Receipts werden ebenfalls gespeichert. Fehlt ein vollständiger Victim-Stream, ist die aggregierte Primärmetrik `null`; die Episode bleibt mit Fehlerstatus erhalten. Batchweise empfangene Tokens behalten ihren tatsächlichen gemeinsamen Empfangszeitstempel.

`measurement_valid` prüft technische Vollständigkeit. `eligible_for_planned_comparison` verlangt zusätzlich, dass alle vier Victims am tatsächlichen Trigger bereits Token 1 und noch nicht Token 64 geliefert hatten. Diese Prüfung beeinflusst weder Timer noch Versand. Ein zu früher oder zu später Pilottrigger ist eine Diagnose, keine nachträglich reparierte Messung.

## Instrumentierung und Prüfbarkeit

Jede Episode erhält ein eigenes exklusiv angelegtes JSONL-Journal mit Plan, Runtime-/Startup-Nachweisen, vollständigen Request-Payloads, rohen SSE-Daten, Token-IDs, Receipt-Zeitstempeln, Fehlern und Abschlussbewertung. Die Auswertung rekonstruiert Token-Endpunkte aus den rohen SSE-Daten und prüft die Kopien gegen das Journal. Abgebrochene Journale werden durch `audit` als unvollständig ausgewiesen; keine angegebene Episode wird still verworfen. Der Blockbericht listet auch nicht gestartete beziehungsweise nicht abgeschlossene Episoden.

Für jeden Request werden Task-/Executor-Zeitstempel, geplante Deadline und tatsächlicher HTTP-Sendehook gespeichert. Der HTTP-Hook liegt nach Request-/Client-Vorbereitung und vor dem ausgehenden Sendepfad; Verbindungsaufbau und Transportlatenz bleiben Teil des Requests. Er ist kein serverseitiger Empfangszeitstempel. Die originalen Transportmethoden verwenden einen gemeinsamen HTTPX-Pool mit erhaltenen individuellen Timeouts. Dadurch entsteht die Burst-Form nicht durch vier nacheinander erzeugte Client-/SSL-Kontexte.

Ein begrenzter einzelner Schreibworker flusht und synchronisiert jedes Journalereignis. Der Timer und die SSE-Empfangsseite warten nicht auf Platten-I/O. Ein Platten- oder Queuefehler verhindert einen erfolgreichen Abschluss. Ein Absturz vor dem endgültigen Flush kann ein unvollständiges Journal hinterlassen; dieses ist ungültig und wird nicht fortgesetzt.

Prometheus wird vor den Victims einmal vollständig erfasst, danach kontinuierlich im 50-ms-Abstand und nach allen Requests noch einmal vollständig. Es gibt nur einen laufenden Scraper. Alle Samples, auch Fehler, bleiben erhalten. Vorher-/Nachher-Samples müssen idle sein, zur API-Prozessstartzeit passen und die komplette Episode einschließlich Timer abdecken. Lücken über 250 ms, Zählerresets, fremde Request-Beiträge und Preemptions machen den technischen Vergleich ungültig.

Startup-Log und effektive API-/Engine-Konfiguration müssen zum Budget und dem laufenden API-Prozess passen. Die Iterationslogs werden als vollständiges Byte-Segment gespeichert. Die unabhängige Prüfung verlangt lückenlose Iterationsnummern, Kontextbilanz gleich der geplanten uncached Inputmenge und kein Überschreiten von `context_tokens + generation_tokens <= budget`. Sie berichtet Budgetauslastung und gemeinsame Prefill-/Decode-Iterationen. Die Logs belegen aggregierte Scheduler-Arbeit; sie liefern keine exakte GPU-Kernelzeit und keine eindeutige Zuordnung jedes Scheduler-Schritts zu einer Request-ID.

Die erweiterte Konfigurationsprüfung erfasst vor und nach jeder Episode `/server_info?config_format=json` vollständig. Der Launcher setzt dafür `VLLM_SERVER_DEV_MODE=1` am Loopback-Server. Fehlende, fehlerhafte oder abweichende Antworten sind ein technischer Fehler; eine fehlende Vorher-Konfiguration verhindert sämtliche Victim-Requests. Beide Konfigurationskopien müssen identisch sein und mit den Journalereignissen übereinstimmen.

Geprüft werden insbesondere Partial-/Long-Prefill-Limits und Schwelle, FCFS, Async-Scheduling, Standard-Scheduler, Stream-Intervall, Hybrid-KV-Manager, Chunking, Sequenzzahl/-länge, KV-Datentyp/-Kapazität, GPU-Speicheranteil, Swap und beide Offload-Pfade sowie TP/PP/DP/Context-Parallelisierung, Executor-Backend, DBO und Expert-Parallelisierung. GPU 0 muss eine RTX 3090 mit 24 GiB sein. BF16, Quantisierung/Speculation/LoRA/KV-Transfer, FLASH_ATTN, Eager-/Compile-/CUDA-Graph-Konfiguration und Iterationslogging werden ebenfalls geprüft. Ausgelassene CLI-Werte werden ausschließlich mit den bekannten vLLM-0.17.1-Defaults ergänzt und anschließend gegen die aufgelöste Konfiguration geprüft. `max_num_scheduled_tokens=None` ist dort der reguläre Fallback auf das geprüfte Tokenbudget.

Alte Journale werden nicht umgeschrieben. Ihnen fehlen die neuen vollständigen Konfigurationssnapshots; der aktuelle striktere Audit kann diesen zusätzlichen Nachweis daher nicht nachträglich bestätigen.

## Offline-Prüfung

Die Tests benötigen nur die Python-Standardbibliothek und den vorhandenen historischen Executor. Empfohlen: Python 3.12 oder 3.13. Für echtes Serving verlangt vLLM 0.17.1 Python <3.14 und seine passende Torch-/CUDA-Umgebung.

```bash
cd /home/rock/projects/DOS
python3 -B -m unittest discover -s new/scripts/prefill_shape_v1/tests -v
python3 new/scripts/prefill_shape_v1/launch_server.py --budget 2048 --log /tmp/shape-unused.log --dry-run
```

`--dry-run` startet keinen Server und legt die Logdatei nicht an. Die Tests verwenden den echten eingefrorenen Executor mit Fake-Transport und verbieten Netzwerkverbindungen. Fake-Endpunkte sind keine wissenschaftlichen Ergebnisse.

## Genau einen Diagnoseblock vorbereiten

Der mitgelieferte Plan `prepared/diagnostic_b2048.json` enthält genau einen randomisierten Dreierblock für Budget 2048: insgesamt 20 Completion-Requests. Die Vorbereitung sendet keine Requests. Der voreingestellte öffentliche Token-Pool umfasst IDs 100–131 des normalen Llama-Tokenvokabulars; vor echtem Serving muss seine Gültigkeit für den lokalen Modell-Snapshot geprüft werden. Eigene öffentliche Token-IDs können bei der Planung explizit angegeben werden.

Eine **neue** Diagnoseplanung lässt sich bei Bedarf exklusiv speichern:

```bash
python3 new/scripts/prefill_shape_v1/run.py prepare --mode diagnostic --output /tmp/new_shape_diagnostic.json
```

Der zusätzliche Plan `prepared/diagnostic_b512.json` enthält einen eigenständigen 512-Diagnoseblock. Neu erzeugen (exklusiver Dateipfad):

```bash
python3 new/scripts/prefill_shape_v1/run.py prepare --mode diagnostic --budget 512 --output /tmp/new_shape_diagnostic_512.json
```

## Manueller 512-Diagnoseblock

In der vLLM-0.17.1-Umgebung und mit identischem `VLLM_API_KEY` in beiden Terminals zunächst einen neuen Server starten:

```bash
cd /home/rock/projects/DOS
python new/scripts/prefill_shape_v1/launch_server.py --budget 512 \
  --log new/logs/prefill_shape_v1/server_shape_v1_diagnostic_b512.log
```

Nach erfolgreichem Startup im zweiten Terminal:

```bash
cd /home/rock/projects/DOS
python new/scripts/prefill_shape_v1/run.py run-block \
  --plan new/scripts/prefill_shape_v1/prepared/diagnostic_b512.json \
  --budget 512 --block 0 \
  --server-log new/logs/prefill_shape_v1/server_shape_v1_diagnostic_b512.log \
  --output-dir new/runs/prefill_shape_v1/diagnostic_b512_checked
```

Es wird kein Hauptkampagnenmodus verwendet. Vorhandene Pfade werden nicht überschrieben.

## Manuelle Startbefehle für den vorbereiteten Block

Voraussetzungen: funktionierende vLLM-0.17.1-Umgebung mit HTTPX, gecachtes Modell, freie RTX 3090, Loopback-Port 8018. `VLLM_API_KEY` muss in beiden Terminals identisch gesetzt sein; der Wert wird nicht als Kommandoargument oder im Journal gespeichert. `python` bezeichnet hier den Interpreter dieser Serving-Umgebung.

Terminal 1:

```bash
cd /home/rock/projects/DOS
python new/scripts/prefill_shape_v1/launch_server.py --budget 2048 \
  --log new/logs/prefill_shape_v1/server_shape_v1_diagnostic_b2048.log
```

Nach vollständig erfolgreichem Startup, Terminal 2:

```bash
cd /home/rock/projects/DOS
python new/scripts/prefill_shape_v1/run.py run-block \
  --plan new/scripts/prefill_shape_v1/prepared/diagnostic_b2048.json \
  --budget 2048 --block 0 \
  --server-log new/logs/prefill_shape_v1/server_shape_v1_diagnostic_b2048.log \
  --output-dir new/runs/prefill_shape_v1/diagnostic_shape_v1_prepared
```

Danach unabhängig auswerten:

```bash
python new/scripts/prefill_shape_v1/run.py audit \
  new/runs/prefill_shape_v1/diagnostic_shape_v1_prepared/*.jsonl
```

Bestehende Logs und Outputverzeichnisse werden nicht überschrieben. Der Runner startet keinen Server; Terminal 1 wird anschließend separat beendet. Es existiert kein automatisch ausgeführter Hauptkampagnen-Loop.

## Offene technische Gates vor einer Hauptkampagne

1. Timer-/Sendelatenz jeweils höchstens 50 ms; tatsächlicher Burst deutlich konzentrierter als Verteilt. Die Deadline allein belegt keine tatsächliche Lastform.
2. Alle Victims decodieren am Trigger; späte zusätzliche Requests und das natürliche Ende einzelner Victims müssen im Diagnoseverlauf geprüft werden, bevor die Hauptparameter eingefroren werden.
3. Lückenlose SSE-/Metrik-/Scheduler-Daten, gültige Vorher-/Nachher-Grenzen, keine fremden Requests oder Preemptions. Konstante KV-Kapazität und unveränderte effektive Nuisance-Konfiguration.
4. Die Budgets müssen in den realen Scheduler-Trajektorien tatsächlich binden. Ein unterschiedlicher Flagwert allein belegt keine Budgetwirkung. Iterationslogging und Client-Journaling können messbaren Overhead verursachen.
5. Ein Diagnoseblock bei 2048 prüft Messbarkeit und Kontrollen; er identifiziert keine Interaktion zwischen 512 und 2048. Haupt-Timer, Haupt-Abstand, Reihenfolge der Budget-Sessions und Auswertungs-/Unsicherheitsverfahren sind vor der Hauptkampagne gesondert festzulegen.

Der neue Runner hat keine bestehenden Prefill-Shape-Rohdaten als eigenen Befund übernommen. Ohne neue reale Messungen wird keine wissenschaftliche Wirkung behauptet.
