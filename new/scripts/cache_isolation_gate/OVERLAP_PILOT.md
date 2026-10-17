# Minimaler Overlap-Pilot – technische Abnahme, 2026-10-17

Implementiert und ausschließlich offline getestet. Kein Serverstart,
GPU-Test, Download oder Netzwerkzugriff wurde ausgeführt. Die zuvor
berichteten 24/64-Ergebnisse werden hier als Ausgangsbefund des Auftrags
geführt; dieses Paket erzeugt keinen neuen empirischen Nachweis.

## Änderungen und Wiederverwendung

Neu sind `overlap_pilot.py`, `overlap_contract.py` und
`tests/test_overlap_pilot.py`. Erweitert sind `record_metrics.py`
(asynchroner Recorder mit demselben Rohsample-Format), `run_server.sh`
(Prompt-Token-Details und Schutz vorhandener Logs), die bestehenden
Starter-Tests sowie diese Dokumentation und die README.

Unverändert wiederverwendet werden der historische Raw-SSE-Executor
`new/scripts/prefill_confirmation/run_prefill_confirmation.py`, sein
`on_output_tokens`-Callback, seine Protokoll-/Timeout-Prüfung und
`cancel_all`; außerdem `load_base` mit SHA-Prüfung, `JsonlRecord`,
Server-/Cache-Konfigurationsprüfungen, `finalize_existing_sampler`,
`check_stream`, `request_identities`, `receipt_at_token`, `parse_metrics`,
`parse_text` und `clock_domain`. Der zwecks 2-Token-Smoke begrenzte
`SaltTransport` wird für die längeren Requests durch einen dünnen
auf diese Episode begrenzten Transportadapter ergänzt.

Historischer Executor-SHA256:
`981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75`.

## Episode und Rohdaten

Ein `run`-Aufruf sendet höchstens D und B, niemals eine Kampagne.
D hat 192 Prompt-/32 Output-Tokens, B 128/64. Ausgangspunkt ist die
vorhandene öffentliche 128-ID-Datei. D ergänzt deren erste 64 IDs.
No-prefix-B verwendet einen deterministisch abweichenden ersten
16er-Block aus `build_prompts`; dieselben B-IDs gelten für beide Salts.
Seeds und Sampling sind fest. Fünf explizite Bedingungen sind verfügbar:
`victim-only`, `shared`, `isolated`, `no-prefix-shared`, `no-prefix-isolated`.

Die Kohorte `[0]`, Identitäten, Tokenlisten, Hashes und Salts stehen vor
dem ersten Request im exklusiv angelegten JSONL. Jede Episode bekommt
einen neuen UUID-Namespace für ihre Salts; es gibt keinen impliziten
Cache-Reset. D muss anschließend tatsächlich 0 Treffer bestätigen.
Restbestände anderer Namespaces sind bei initialem Idle evictierbar;
die physische Auslastung wird mit aufgezeichnet.

Beim achten empfangenen D-Output-Token erzeugt der vorhandene Executor
den synchronen Callback. Dieser legt genau eine B-Task an. Ein SSE-Batch,
der von weniger als 8 auf mehr als 8 Tokens springt, macht den Trigger
ungültig und startet B nicht. D wird bis zum vollständigen Abschluss mit
32 Tokens gelesen, B mit 64; D28 ist kein Abbruchpunkt.

Jedes rohe SSE-Ereignis wird sofort mit `flush` und `fsync` journalisiert.
Der Adapter erhält den echten `perf_counter_ns`-Empfangszeitpunkt vor
diesem Schreiben und gibt genau diesen an den eingefrorenen Executor
weiter. Damit stimmen Wire-Journal und dessen SSE-Zeitstempel überein.
Task-Erzeugung, tatsächlicher Dispatch und Trigger-Beobachtung haben
eigene Zeitstempel; die Dispatch-Verzögerung wird ausgewiesen.

Alle Metrik-Scrapes enthalten GET-Start, Antwort-Empfang, HTTP-Status,
Fehler und den vollständigen Rohtext. Der Recorder läuft durch beide
Streams hindurch und wartet danach begrenzt auf abgeschlossene Zähler
und Idle. Der Abschlussadapter übernimmt seine vollständige Historie.
Fehler lassen D und B in der deklarierten Kohorte; nicht gestartete
Requests bekommen `not_dispatched`. Ein partielles Journal ohne
Enddatensatz ist ungültig und bleibt erhalten.

Der vollständige Startup-Text, API-PID, Linux-Prozessstart, Boot-ID,
Datei-/Implementierungshashes und der unveränderte Executor-Hash werden
gespeichert. Die PID/Startzeit werden gegen `/proc` und jede Live-Metrik
geprüft. Das Episodenlog wird mit Byte-Grenzen und SHA gespeichert;
Ersetzen oder Abschneiden der Quelldatei macht die Episode ungültig.
Der Offline-Reader gleicht Enddaten mit dem dauerhaften Journal ab.

## Gates und ihre Grenzen

`technical_complete` prüft vollständige Requests, Identitäten, Salts und
Workload. `trigger_exact` prüft den Callback gegen die Roh-SSE und D8..28.
`server_bound` prüft Prozessbindung und unveränderte effektive Konfiguration.
`metrics_complete` verlangt Samples vor dem Request, vollständig innerhalb
D8..28 und nach beiden Abschlüssen, ohne Reset oder Erfassungslücke.
Ohne vollständige Zuordnung gibt es keinen numerischen Primärendpunkt.

`cache_manipulation` verlangt D=0 und B=112 nur für `shared`, andernfalls
B=0. Der neue Starter aktiviert `--enable-prompt-tokens-details`.
vLLM 0.17.1 lässt diese Details bei 0 Hits trotzdem weg: Das wird nur
bei bestätigtem aktivem Flag als 0 gelesen und mit den globalen Zählern
abgeglichen. Queries/Prompt-Tokens, abgeschlossene Prefills und
Running/Waiting dienen zusätzlich der Fremdverkehrsprüfung.

Neue Prefill-Tokens je Request werden aus Prompt minus bestätigten Hits
abgeleitet und mit dem Histogramm-Zähler abgeglichen. Das Histogramm
selbst ist kein vollständiger Recompute-Zähler. Deshalb werden zusätzlich
alle Scheduler-Kontext-Tokens, Decode-Tokens und Chunks in ihrer Reihenfolge
gespeichert. Waiting, Preemption und KV-Usage bleiben als Rohzeitreihen
erhalten. Metrik-Antwortzeit ist kein exakter Ereigniszeitpunkt im Scheduler.

`iteration_accounting` verlangt lückenlose Iterationen, vollständig
abgerechnete Prefill-/Decode-Arbeit und einen Token pro Decode-Request.
`effective_budget` verlangt Startup-Bestätigung und Einhaltung pro
Iteration. `budget_binding_observed` bezieht sich bei D/B auf B-Prefill;
ein bindendes Budget allein während des anfänglichen D-Prefills genügt
nicht. Der Offline-Budgetkontrast verlangt passende Baselines, gleiche
Workload/Cache-Zustände und tatsächlich verschiedene B-Chunks bei 64/512.
Insbesondere kann B mit 112 Hits und nur 16 neuen Prefill-Tokens bei
beiden Budgets nicht bindend bleiben; das ist ein zulässiges negatives Gate.

Für `overlap_confirmed` genügt überlappende Client-Lebensdauer nicht:
Mindestens ein B-Token muss innerhalb D8..28 eintreffen. Zusätzlich müssen
mehr als vier Iterationen D und B gemeinsam schedulen. Ohne Fremdverkehr,
Preemption und Speculation können höchstens vier D-Decode-Schritte nach
D28 liegen; damit ist mindestens ein gemeinsamer Schritt im Zielintervall
belegt. Weniger gemeinsame Schritte bleiben unbestätigt. Es werden keine
Nanosekunden aus den nur sekundengenauen Scheduler-Logzeiten erfunden.

Bei Preemption bleiben technisch vollständige Streams und die Beobachtung
erhalten. Seit der Nachbesserung E2 werden drei Aussagen getrennt:
`technical_sse_measurement_complete` bestätigt vollständige SSE samt Trigger;
`mechanism_status=preemption_candidate_insufficient_attribution` kennzeichnet
einen an die Serverinstanz gebundenen, monoton gemessenen Preemption-Befund;
`causal_comparison_eligible` bleibt bei unzureichender Zuordnung `false`.
Die einfache Cache-/Iterationszuordnung und das Overlap-Gate werden damit
weiterhin nicht ungeprüft bestätigt. Dafür wären zusätzliche Scheduler-
Identitäten nötig. Dieses Paket verändert dafür keinen vLLM-Code.

Die vollständige Messung kann dann `raw_endpoint_ms` liefern, während
`endpoint_ms` für den kontrollierten Vergleich leer bleibt. `valid_episode`
bleibt aus Kompatibilitätsgründen das strenge Zulässigkeitsgate.
`mechanism_candidate` und die Kandidatenliste des Offline-Vergleichs erhalten
die Beobachtung ausdrücklich. Ein Preemption-Zähler ohne belastbare
Server-/Metrikbindung heißt `unattributed_preemption_counter_change`.
Ein regulärer Kandidat verlangt zusätzlich `traffic_attributed=True`.
Bei erkanntem Fremdverkehr oder unklarer Verkehrszuordnung lautet der Status
`unattributed_preemption_foreign_or_ambiguous_traffic`; `mechanism_candidate`
bleibt dann `false`, die gemessene Preemption und vollständige SSE bleiben
erhalten. Der Exitcode ist in diesem Fall 1 statt 3.
`h1_status=not_evaluated` gilt für alle Episoden: Ein unbestätigtes Gate
ist keine falsifizierte H1 und ein Kandidat kein H1-Nachweis.

Kapazitätsbedingungen verlangen eine gültige Victim-only-Baseline derselben
Konfiguration ohne Waiting/Preemption. Als Drucksignal gelten Preemption
oder mindestens 95 % gemeldete aktive KV-Auslastung. Der Kapazitätskontrast
verlangt dieses Signal bei 24 und dessen Ausbleiben bei 64, zusätzlich zu
gültigen Episoden und einer höheren beobachteten absoluten aktiven
Blockbelegung bei 64. Eine andere Prozentzahl allein genügt nicht; die
aktive Blockzahl folgt aus Usage × (Blöcke − 1), siehe `BlockPool.get_usage`.
Die Kandidaten allein bestätigen den Kontrast nicht.
Eine verpasste kurze Auslastungsspitze führt zu einem unbestätigten Gate.

Vergleiche prüfen die Victim-Identität und den vollständigen effektiven
Engine-/API-Konfigurationsfingerprint. Seit der Nachbesserung E1 wird die
Engine-Repräsentation feldweise statt als unveränderliche Gesamtzeichenkette
verglichen. Nur die auf den deklarierten Wert geprüften Budget-/KV-Felder
werden normalisiert: `max_num_batched_tokens`, `num_gpu_blocks_override`
und `num_gpu_blocks`, auch in `SchedulerConfig`/`CacheConfig`.
vLLM 0.17.1 setzt außerdem das letzte `compile_ranges_split_points` auf das
Budget (`config/vllm.py:1492`); ausschließlich diese geprüfte Endgrenze wird
ebenfalls normalisiert. Alle anderen Splitpunkte, Capture-Größen, Seeds,
Datentypen, Listener-Adressen und unbekannten Felder bleiben vergleichspflichtig.
Die vollständige ursprüngliche Engine-Zeichenkette bleibt im Rohlog erhalten.
Falsche Faktorwerte werden abgewiesen, nicht durch Platzhalter verdeckt.
Eine Ausnahme betrifft ausschließlich den noch nicht initialisierten
Istwert `num_gpu_blocks=None` im frühen Engine-Log: vLLM 0.17.1 protokolliert
die Konfiguration vor `_initialize_kv_caches` und setzt den Istwert danach
(`v1/engine/core.py:101` und `:124`). Der Override muss weiterhin passen.
Die späteren KV-Kapazitätsmeldungen und Live-Metriken müssen die konkrete
Blockzahl bestätigen; dort wird `None` nicht akzeptiert.
Salt-Kontraste weisen die Verfügbarkeit beider No-prefix-Kontrollen extra
aus. Das Programm berechnet keine Latenzeffekte zwischen Bedingungen und
setzt H1-/Neuheitsfelder immer auf `false`.

## Kapazitätsrechnung und verbleibende Confounder

D benötigt höchstens `ceil((192+32)/16)=14`, B höchstens
`ceil((128+64)/16)=12` aktive Blöcke. Mit 7 geteilten Trefferblöcken sind
es gemeinsam höchstens 19, ohne Reuse höchstens 26. Nach dem reservierten
Null-Block bleiben bei 24/64 Blöcken 23/63 verfügbar. Diese Obergrenzen
garantieren keine gleichzeitige Belegung: D kann vor dem B-Wachstum enden.
Deshalb bleibt der Kapazitätskontrast ein zu beobachtendes Runtime-Gate.

Die wesentlichen offenen Risiken sind SSE-Bündelung und Pufferung,
Dispatch-Verzögerung durch Journaling, Mess- und Logging-Overhead,
verzögerte/grobe Metriken, unbemerkte Last anderer Clients sowie unpassende
Compilation-/Warmup-Zustände. No-prefix ist eine Kontrolle der Salt-Zuordnung,
keine vollständige Matched-Compute-Erklärung. Ein positiver Latenzkontrast
kann weiterhin durch gewöhnliche zusätzliche Prefill-Arbeit erklärt sein.

Die Codeableitungen wurden lokal in vLLM 0.17.1 geprüft:
`entrypoints/openai/completion/serving.py:452` (Cache-Details),
`v1/metrics/loggers.py:1163` (Prefill-Histogramm),
`v1/utils.py:436` (`compute_iteration_details`),
`v1/engine/core.py:349` (`log_iteration_details`) und
`v1/core/block_pool.py:487` (`get_usage`, Null-Block).

## Offline-Abnahme und kleinster erster GPU-Pilot

Reproduzierbare Offline-Befehle ab Repository-Wurzel:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s new/scripts/cache_isolation_gate/tests -v
bash -n new/scripts/cache_isolation_gate/run_server.sh
python3 -B new/scripts/cache_isolation_gate/verify_sources.py --root "$PWD"
```

Abnahmestand nach den beiden Folgekorrekturen: 85 Tests bestanden, keine fehlgeschlagenen Tests. Shell-Syntax
und die beiden historischen Quellenhashes wurden ebenfalls geprüft. Die
negativen Fixtures sind erwartete Zurückweisungen innerhalb bestandener Tests.

Die Suite prüft alle fünf Typen, frühe/späte/mehrfache Trigger, gebündelte,
fehlende und verspätete SSE, Identitäts-/Salt-/Cache-Fehler, Fremdverkehr,
Preemption, Timeout, fehlende Post-Trigger-Metriken, nicht bindendes Budget,
falsche Log-/Ergebniszuordnung und vorhandene Ausgabedateien. Die
Integrationstests verwenden den echten historischen Executor, einen Fake-
Transport und blockieren `socket.connect`. Es gibt keine GPU-Testdaten.

Die gezielten E1-Regressionen benutzen unterschiedliche Engine-Logs für
64/512 und 24/64, einschließlich verschachtelter Config-/Enum-Repräsentationen
und der abgeleiteten Compile-Grenze. Einzelne Episoden und passende
Faktorvergleiche bestehen; zusätzliche Änderungen an Datentyp oder
Compilation-Einstellungen scheitern weiterhin am Vergleich. E2 prüft eine
einzelne Preemption bei vollständiger Messung als erhaltenen Kandidaten,
mit fehlender Zuordnung und `h1_status=not_evaluated`. Unvollständige SSE
oder falsche Serverbindung werden dadurch nicht akzeptiert.
Die Folgekorrekturen prüfen den frühen `None`-Istwert bei 24 und 64 Blöcken,
die weiter strikte spätere Startup-/Live-Prüfung sowie Fremdverkehr und
Preemption gemeinsam (zusätzlicher abgeschlossener oder laufender Request).

**NO-GO für den GPU-Start bis zur erneuten Prüfung dieser beiden Korrekturen.**
Als nächster ausdrücklich freizugebender technischer Block ist vorgesehen:
64 GPU-Blöcke, Modelllänge 256, Budget 512, Offload 0. Zuerst eine gültige
Victim-only-Episode; dann je eine Shared-, Isolated- und beide No-prefix-
Episoden mit derselben Token-Datei und eigenen Ergebnisnamen. Das sind
fünf Episoden/neun Requests und keine Acht-Zellen-Kampagne. Der Block prüft
erst die Manipulation und Erfassung; Budget-/Kapazitätskontraste bleiben
unbestätigt. Bei einem ungültigen Gate stoppen und Rohdaten prüfen.

Erst danach wären separat freigegebene, passende Baseline-/D/B-Aufrufe bei
Budget 64 bzw. Kapazität 24 sinnvoll. Fehlende B-Budgetbindung, fehlender
Kapazitätsdruck oder ein Scheitern der negativen Kontrolle beendet die
jeweilige Interpretation. Es gibt keinen automatischen Wiederholungs- oder
Kalibrierungsmodus.

Beispiel für **einen später freizugebenden** Aufruf gegen einen bereits mit
dem aktualisierten Starter gestarteten und lokal authentifizierten Server:

```bash
python3 -B new/scripts/cache_isolation_gate/overlap_pilot.py run \
  --root /home/rock/projects/DOS \
  --prompt-token-ids new/runs/cache_isolation_gate/public_prompt_token_ids_128.json \
  --condition victim-only --server-log PFAD_ZUM_AKTUELLEN_LOG \
  --blocks 64 --budget 512 \
  --output new/runs/cache_isolation_gate/overlap_EINDEUTIGER_NAME.jsonl
```

`VLLM_API_KEY` muss in derselben bereits vorhandenen vLLM-0.17.1-Umgebung
gesetzt sein. Die anderen Bedingungen werden einzeln gewählt. Der reine
Offline-Aufruf ist `python3 -B new/scripts/cache_isolation_gate/overlap_pilot.py
audit DATEI1.jsonl DATEI2.jsonl ...`. Exit 0 von `run`/`audit` bestätigt nur
gültige einzelne Episoden; die getrennten Kontrast-Gates müssen zusätzlich
geprüft werden. Exit 3 kennzeichnet vollständig gemessene, erhaltene
Mechanismenkandidaten mit unzureichender Zuordnung. Exit 1 kennzeichnet
unvollständige Messungen oder andere verletzte Kontrollgates, auch wenn
zusätzlich ein Kandidat gespeichert wurde. Die JSON-Statusfelder sind
maßgeblich. Es wird keine Genehmigung aus einem Exit-Code abgeleitet.
