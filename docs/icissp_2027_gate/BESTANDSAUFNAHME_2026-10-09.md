# NordSec → ICISSP 2027: technische Bestandsaufnahme und erster Workflow

Stand: 9. Oktober 2026. RDNA / Audit Mode, Z2. Dieser Bericht ersetzt den anfänglichen Workflow mit dem damals noch ungeklärten Repositoryzugriff.

**Entscheidung: Die vorhandene Messbasis ist wiederverwendbar. Die statische Bestandsaufnahme ist abgeschlossen; das technische Laufzeitgate für die acht Zellen ist noch offen.** Es wurde kein neuer Messrequest gesendet, kein Modell geladen und keine Kampagne gestartet. Eine Interaktion oder Schutzumkehr wird nicht als Ergebnis behauptet.

## 1. Tatsächlich geprüfter Stand

| Bereich | Befund |
| --- | --- |
| Erreichbarer Checkout | `/home/rock/projects/DOS`, Commit `d1e247ba22b3c505f3b36173655b937f89c2693f` vom 6. August 2026; Git-Arbeitsstand vor und nach der Prüfung sauber. |
| GitHub-Stand | `TheBuccaneer/DOS`, `main` bei der Abfrage: `aee916be70765334218449a569d7629cb69672d7` vom 9. Oktober 2026. Weitere Lesezugriffe wurden auf diesen Commit fixiert. |
| Vergleich der Quellen | Sechs relevante Dateien im Checkout sind über ihre Git-Blob-Hashes identisch mit dem geprüften GitHub-Stand: Root-Serverstarter, Budget-Serverstarter, Budget-Runner, Prefill-Basisrunner, Server-WAITING-Runner und `_active_cohort.py`. |
| Bereits vorhandene Vorbereitung | GitHub enthält `ICISSP_2027_Vorbereitung_2026-10-09/` mit Workflow, Offline-Auditor, Umgebungsprüfung und einem ausdrücklich noch nicht ausführbaren Acht-Zellen-Entwurf. Diese Dateien fehlen im älteren Checkout. Ihre Kernbefunde wurden erneut geprüft. |
| Geeignete Python-Umgebung | `/home/rock/projects/porto/.venv-py313-repair/bin/python3`: Python 3.13.7, vLLM 0.17.1, Torch 2.10.0, Transformers 4.57.6, httpx 0.28.1, NumPy 2.2.6, Pandas 3.0.1 und SciPy 1.17.1. |
| Andere historische Umgebung | `/home/rock/projects/porto/.venv/bin/python3` ist vorhanden, liefert hier aber Python 3.14.4 und keine der geprüften Messabhängigkeiten. Sie ist aktuell kein bestätigter Ersatz. |
| Installierter vLLM-Code | Scheduler, Scheduler-Konfiguration, Completion-Protokoll und Metriklogger der Reparaturumgebung sind bytegleich mit den abgerufenen offiziellen v0.17.1-Dateien. |
| Modellbestand | Llama-3.1-8B-Instruct, Revision `0e9e39f249a16976918f6564b8830bc894c89659`, und Qwen2.5-7B-Instruct, Revision `a09a35458c702b33eeacc393d103063234e8bc28`: Konfiguration und je vier sichtbare Gewichtsshards vorhanden. Vollständigkeit und Ladbarkeit wurden nicht durch einen Modellstart bestätigt. |
| GPU-Zugriff | `nvidia-smi` ist vorhanden, scheitert aber mit Exit-Code 9 an der Kommunikation mit dem NVIDIA-Treiber. Die Ursache wurde nicht diagnostiziert. Historische Manifeste belegen eine RTX 3090; sie ersetzen keine aktuelle Funktionsprüfung. |

Die Root-Berichte vom 6. Oktober und `new/README.md` dienen der Einordnung. Das README beschreibt eine ältere Offload-Matrix. NordSec-Manuskript, Originalreviews und FairCache-Volltextabgleich wurden in den geprüften Root-/`new`-/Dokumentpfaden nicht lokalisiert. Das blockiert die spätere Claim-Prüfung, nicht diese technische Bestandsaufnahme.

## 2. Wiederverwendbare Komponenten

| Komponente | Wiederverwendbarer Teil | Erforderliche Anpassung |
| --- | --- | --- |
| `new/scripts/prefill_confirmation/run_prefill_confirmation.py` | Streaming mit Token-IDs, Raw-SSE, monotonen Empfangszeiten, Prompt-/Seed-Identitäten und Zeitgrenzen. | Der aktuelle Request-Payload enthält kein `cache_salt`. Den historischen Basisrunner als Referenz erhalten; eine gezielte Erweiterung für die neue Studie vorsehen. |
| `new/scripts/chunk_budget_screen/` | Explizite Budgetübergabe, gepaarte Läufe, ausbalancierte Reihenfolge und Provenienz. | Prefix-Caching ist ausgeschaltet; Cache-Regel und kontrollierte KV-Kapazität fehlen als Faktoren. |
| `new/scripts/server_waiting_confirmation/` und Llama-Port | Persistenter Completion-Transport, separater Metriktransport und Hintergrundsampler. | Vollständige Metrikzeitreihe speichern und dynamische Kohortenwahl für den neuen Paarendpunkt ersetzen. |
| `_active_cohort.py` | Tokenfortschritt und Triggerrekonstruktion. | Vorab bestimmte Request-Identitäten verwenden; keine Auswahl nach beobachteter Fertigstellungsreihenfolge. |
| Vorhandene Auditoren | Raw-Trace-, Zeit-, Identitäts- und Provenienzprüfungen. | Neue Cache-/Kapazitätsfelder und Endpunktregeln ergänzen. Die strenge Prüfung der historischen Serverkommandos gilt nicht automatisch für einen neuen Starter. |
| Bestehende Root-/Workshop-Auswertung | Datenverständnis und Auswertungsmuster. | Neue primäre Kohorten-Zielzeit und Interaktionsauswertung ergänzen; Altbefunde nicht als neue Studienergebnisse übernehmen. |

Der Root-Starter `run_server.sh` ist keine sichere Vorlage für einen unveränderten Aufruf: Er berechnet seinen Projektpfad über `../..`, obwohl er im Repository-Root liegt. Der kanonische Budget-Starter unter `new/scripts/chunk_budget_screen/` berechnet den Pfad passend zu seinem Ort. Für die neue Studie den kanonischen Starter als Ausgangspunkt verwenden.

## 3. Erneut ausgeführter Offline-Audit

Der bereits vorhandene Auditor aus dem Vorbereitungsordner wurde nach Codeprüfung in einer temporären Arbeitskopie ausgeführt. Er nutzt die Standardbibliothek, importiert keine Experimentrunner und sendet keine Requests.

| Historische Kampagne | Episoden | Paare | Requests | Ausgabetokens | Integritätsfehler |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunk Budget, Juli | 36 | 18 | 792 | 47.232 | 0 |
| Server WAITING, Llama | 32 | 16 | 704 | 41.984 | 0 |
| Server WAITING, Qwen | 32 | 16 | 704 | 41.984 | 0 |
| Summe | 100 | 50 | 2.200 | 131.200 | 0 |

Geprüft: Rekonstruktion der Ausgabetoken aus SSE, Tokenzahlen, monotone Empfangszeiten, erster/letzter Tokenzeitpunkt, Request-Zeitgrenzen, Identität der Decode-Population innerhalb der Paare und Paarreihenfolge. Alle Episoden sind als vollständig markiert; in jeder Kampagne ist die Reihenfolge exakt ausbalanciert. Dies ist ein begrenzter Integritätscheck, keine vollständige wissenschaftliche Validierung der Altstudie.

**Zwei bestätigte Designlücken:** In den 64 Server-WAITING-Episoden wurden 21.647 Metriksamples gespeichert, davon kein einziges nach dem logischen Trigger. In allen 32 zugehörigen Paaren unterscheiden sich die dynamisch ausgewählten aktiven Kohorten, obwohl die gesamte angebotene Decode-Population innerhalb der Paare identisch ist.

## 4. Cache- und Budgetparameter

| Parameter | Bestätigte Bedeutung im geprüften Stand | Konsequenz |
| --- | --- | --- |
| `enable_prefix_caching` | Unterstützt; relevante historische Starter setzen explizit `--no-enable-prefix-caching`. | Caching muss für beide neuen Policy-Stufen aktiviert sein. Shared/Isolated ist ein Wiederverwendungsfaktor, kein Vergleich Caching an/aus. |
| `cache_salt` | Completion-Protokoll akzeptiert nichtleere Strings; Serving gibt den Salt an den Renderer weiter, der Input-Prozessor übernimmt ihn, und die KV-Hashbildung berücksichtigt ihn im ersten Block. | Stabiler Salt pro Tenant. Ein Salt pro Request würde auch tenantinterne Wiederverwendung verhindern. Salts trennen Wiederverwendung, reservieren aber keine tenantweisen Speicherquoten. Laufzeitwirkung bleibt zu testen. |
| `max_num_batched_tokens` | Gesamtbudget je Iteration. Der Scheduler verwendet `max_num_scheduled_tokens`, sofern gesetzt, sonst `max_num_batched_tokens`. | Beide effektiven Werte dokumentieren. Ein akzeptierter CLI-Wert allein ist kein Scheduling-Nachweis. |
| `long_prefill_token_threshold` | Ein positiver Wert begrenzt im Scheduler die neuen Tokens eines Requests, bevor das verbleibende Gesamtbudget angewendet wird. | Für den ersten Budgetvergleich konstant und ausdrücklich festlegen. |
| `max_num_partial_prefills` | Kein direkter Zugriff in der geprüften Scheduling-Logik. In der Konfiguration wird bei einem Wert größer als eins und zuvor null gesetztem Threshold automatisch ein positiver Threshold abgeleitet. | Nicht pauschal als wirkungslos behandeln: indirekte Wirkung über den Threshold ist möglich. Zunächst konstant halten. |
| `max_long_partial_prefills` | In Konfiguration/Validierung vorhanden; keine direkte Verwendung in der geprüften Scheduling-Logik gefunden. | Keine selbstständige wirksame Faktorstufe ohne Laufzeitnachweis. |
| `async_scheduling` / `scheduler_cls` | Konfiguration entscheidet zwischen Standard-, Async- oder eigener Scheduler-Klasse. Der installierte AsyncScheduler erbt die Scheduling-Logik. | Tatsächlich aufgelöste Klasse und effektiven Async-Zustand beim neuen Serverstart erfassen; nicht aus alten Flags erraten. |

**Die zwei neuen Budgetstufen sind noch nicht abgenommen.** Historische Logs bestätigen unterschiedliche konfigurierte Budgets, enthalten aber keinen vollständigen zeitlichen Scheduling-Nachweis für das eingefrorene neue Design.

Ein eingebauter Beobachtungspunkt ist vorhanden: `--enable-logging-iteration-details`. EngineCore protokolliert pro Iteration Context-Requests/-Tokens und Generation-Requests/-Tokens. `compute_iteration_details` leitet diese Werte aus `SchedulerOutput.num_scheduled_tokens` ab und berücksichtigt weitere Prefill-Chunks. Damit ist ein direkter Budgetnachweis voraussichtlich ohne Engine-Patch möglich. Das Logging ist zunächst für Funktionsproben gedacht; sein Messaufwand und seine Vollständigkeit müssen am laufenden Server geprüft werden. Die dort ausgegebene CPU-Zeit ist keine isolierte GPU-Kernelzeit.

## 5. Kapazität und Metriksemantik

Die historischen Llama-Budgetlogs belegen unterschiedliche nutzbare KV-Kapazitäten, auch in Läufen ohne Weight-Offload:

| Historisches Log | Budget | Protokollierte KV-Kapazität |
| --- | ---: | ---: |
| `llama_block01_low_budget512.log` | 512 | 49.952 Tokens |
| `llama_block03_low_budget1024.log` | 1.024 | 42.304 Tokens |
| `llama_block08_low_budget1024.log` | 1.024 | 49.520 Tokens |
| `llama_block05_low_budget2048.log` | 2.048 | 48.656 Tokens |

Das sind Befunde aus Startlogs, keine kontrollierte Kapazitätsintervention. Sie belegen nicht die Ursache der Variation. Die Altbudgetläufe dürfen deshalb nicht unmittelbar als Vergleich bei konstanter KV-Kapazität verwendet werden.

Im installierten Stand sind `kv_cache_memory_bytes` und `num_gpu_blocks_override` einschließlich CLI-Übergabe und ihrer Implementierung vorhanden. Eine explizite Kapazitätsvorgabe ist somit technisch vorgesehen. Für die Studie einen kontrollierten Weg wählen und die tatsächlich resultierende Blockzahl, Blockgröße, KV-Datentyp und Cache-Gruppen nach Serverstart abnehmen. Werte erst nach dem Funktionsnachweis am Messsystem festlegen; gleiche `gpu_memory_utilization` genügt nicht.

| Messgröße | Vorhandener Messpunkt | Einschränkung |
| --- | --- | --- |
| Cache-Hits/-Queries | Prometheus-Zähler für Prefix-Reuse | Zeitliche Differenzen und Engine-/Modelllabels erhalten; Warm-up nicht mit Interventionsdaten vermischen. |
| Ausgeführte Prefill-Arbeit | `prompt_tokens_by_source` und Iterationslogging | Zähler sind aggregiert; Context-Logging liefert zusätzlich den Verlauf. Die angebotene Arbeit bleibt von tatsächlicher Rechenarbeit zu unterscheiden. |
| KV-Belegung | `kv_cache_usage_perc` | Der Wert beruht auf nichtfreien Blöcken im Blockpool, abzüglich des Nullblocks. Er ist keine Messung des gesamten vorreservierten GPU-Speichers und keine Zählung aller weiterhin gecachten, aber wieder freigegebenen Blöcke. |
| Preemptions | `num_preemptions_total` | Kumulativer Engine-Zähler, keine vollständige Requesthistorie. |
| Wiederberechnung | `prompt_tokens_recomputed_total` existiert | Dieser Zähler erfasst hier insbesondere den letzten erneut berechneten Token bei vollständig gecachtem Prompt. Er ist kein Nachweis der durch Preemption wiederberechneten Tokenarbeit. |
| ITL / später Tokenindex | Raw-SSE mit echten Token-IDs | Ein SSE-Event kann mehrere Tokens enthalten; dann gilt ein gemeinsamer Empfangszeitpunkt. |
| Wartezeit / Ausführung | Client-Metadaten und aggregierte Servermetriken | Client-Admission und Dispatch separat behandelbar; serverseitige Requestzeiten ohne weitere Zuordnung nicht für jedes Kohortenmitglied exakt rekonstruierbar. |

## 6. Konkrete minimal notwendige Änderungen

1. **Studienpayload:** Tenant-ID, stabile Cache-Regel und Salt-Zuordnung ergänzen; gesendete Präfixidentität und identische angebotene Tokenarbeit im Manifest prüfen. Anschlussstelle ist die bestehende Completion-Payload-Erzeugung. Den historischen Basisrunner nicht unbemerkt verändern.
2. **Serverstarter:** Kleinen Starter für die neue Studie aus dem kanonischen Budget-Starter ableiten. Prefix-Caching in beiden Regeln an, CPU-Weight-Offload null, Budget und Kapazität explizit. Effektive Konfiguration statt nur Startkommando sichern.
3. **Vollständiger Sampleroutput:** Nach Ende des Samplers sämtliche Samples in einem gesonderten Episodenfeld sichern, auch bei Fehlern. Die vorhandene Trigger-Snapshot-Liste bleibt ein Snapshot. Ursache der aktuellen Lücke: `list(metrics_sampler.samples)` wird beim Trigger kopiert und nach `stop()` nicht ergänzt.
4. **Kohorte und Endpunkt:** Vorab bestimmte Request-Identitäten verwenden. Vorschlag: Zeit bis alle Kohortenmitglieder den späteren Tokenindex erreicht haben. Tokenindex, Beobachtungsursprung, Aggregation und Umgang mit nicht erreichten Indizes vor dem ersten Ergebnis festlegen.
5. **Parser und Auditor:** Vorhandene Rohmetriktexte für KV, Cache und Preemption auswerten; vollständige Zeitabdeckung, Kohortenidentität, effektive Kapazitätsgleichheit und Faktorwirkung prüfen. Historische Kommando-/Hashverträge durch einen eigenen nachvollziehbaren Studienvertrag ergänzen.

Das sind gezielte Erweiterungen der vorhandenen Messkette. Keine neue allgemeine Infrastruktur, kein Scheduler-Neuentwurf und aktuell kein notwendiger Engine-Umbau. Die Änderungen sind hier noch nicht umgesetzt; der DOS-Checkout liegt außerhalb der aktuell freigegebenen Schreibpfade.

## 7. Ausführbare Vorprüfung und erster Diagnoseablauf

Die folgenden bereits vorhandenen Prüfwerkzeuge wurden erfolgreich verwendet. Sobald der aktuelle GitHub-Vorbereitungsordner im Messcheckout vorhanden ist, lassen sich dieselben lesenden Prüfungen dort so wiederholen:

```bash
/home/rock/projects/porto/.venv-py313-repair/bin/python3 /home/rock/projects/DOS/ICISSP_2027_Vorbereitung_2026-10-09/inspect_environment.py --root /home/rock/projects/DOS
```

```bash
/home/rock/projects/porto/.venv-py313-repair/bin/python3 /home/rock/projects/DOS/ICISSP_2027_Vorbereitung_2026-10-09/audit_existing.py --root /home/rock/projects/DOS
```

**Voraussetzung:** Der Vorbereitungsordner ist im hier erreichbaren älteren Checkout noch nicht vorhanden. Bei dieser Prüfung wurde ausschließlich eine temporäre Kopie der versionierten Prüfwerkzeuge ausgeführt. Die Befehle oben sind keine Kampagnenbefehle.

Danach, am funktionsfähigen lokalen Messsystem:

1. GPU-Kommunikation und Modellstart mit der gefundenen Reparaturumgebung prüfen; vorhandenes Modell ohne Weight-Offload verwenden. Nicht aus dem derzeitigen Treiberfehler ungeprüft eine Neuinstallation ableiten.
2. Nach den kleinen Änderungen zunächst Cache-Trennung und vollständige Messaufzeichnung in begrenzten Funktionsproben einzeln abnehmen. Öffentliche Testpräfixe und vorab bestimmte Decode-Identitäten verwenden.
3. Zwei zulässige Budgetstufen bei konstanter Kapazität prüfen. Die Context-Tokenfolgen im Iterationslog müssen sich tatsächlich unterscheiden. Andernfalls ist der Faktor ungültig.
4. Ausreichende und grenznahe Kapazität mit identischer angebotener Arbeit festlegen. Für jeden Policy-/Budgetvergleich identische effektive Blockkapazität abnehmen; noch keinen Kapazitätsmechanismus als bewiesen behandeln.
5. Erst nach bestandenem Gate die acht Zellen ausführen: Shared/Isolated × Small/Large × ausreichende/grenznahe Kapazität. Pro Zelle gepaarte Referenz-/Zusatzlastläufe mit kontrolliertem Cache-Ausgangszustand und ausbalancierter Reihenfolge.
6. Unabhängige Wiederholungen anhand vorab festgelegter Präzision und einer begrenzten Pilotstreuung bestimmen. Tokens oder Requests sind keine unabhängigen Replikate. Keine Verlängerung nach gewünschtem Vorzeichen.

Ein vollständig ausführbarer Live-Diagnoserunner kann noch nicht als fertig gemeldet werden: Seine minimalen Änderungen fehlen und der GPU-Funktionsnachweis ist offen. Nicht bestätigte Parameterwerte werden nicht durch Platzhalterkampagnen ersetzt.

## 8. Gate und weitere Entscheidung

| Kriterium | Status |
| --- | --- |
| Repository, historische Daten und nutzbare Python-Abhängigkeiten zugeordnet | Bestätigt |
| Raw-SSE für festen Tokenindex grundsätzlich geeignet | Bestätigt, SSE-Bündelung beachten |
| Cache-Salt-Pfad und Budgetsemantik im installierten Code vorhanden | Statisch bestätigt |
| Gemeinsame/tenantgetrennte Wiederverwendung tatsächlich beobachtet | Offen |
| Zwei unterschiedliche Scheduling-/Prefill-Verläufe tatsächlich beobachtet | Offen |
| Kapazität über Policy-/Budgetvergleiche konstant | Noch nicht hergestellt |
| Vollständige Interventionsmetriken und vorab feste Kohorte | Anpassung erforderlich |
| RTX-3090-Modelllauf in der aktuellen Umgebung | Nicht bestätigt |

**Nächster Arbeitsschritt:** Die kleinen Änderungen an Payload, Samplerabschluss, Kohortenvertrag und Starter konkretisieren; die Laufzeitabnahme am funktionsfähigen Messsystem durchführen. Keine vollständige Kampagne vor dieser Abnahme.

Für jede Kapazitätsstufe separat gilt weiter `E_p = H_p(small) − H_p(large)` und `Gamma = E_isolated − E_shared`. Eine positive Interaktion ist keine Schutzumkehr. Dafür müssten zusätzlich `E_shared < 0` und `E_isolated > 0` auf derselben primären Metrik mit geeigneter Unsicherheit gestützt sein. H0, H1 und Artefakterklärung bleiben offen. Manuskript und allgemeine Novelty-Suche wurden nicht begonnen.

## Quellen und Reproduzierbarkeit

- [Geprüfter GitHub-Stand](https://github.com/TheBuccaneer/DOS/tree/aee916be70765334218449a569d7629cb69672d7).
- [Bereits vorhandener Vorbereitungsworkflow](https://github.com/TheBuccaneer/DOS/blob/aee916be70765334218449a569d7629cb69672d7/ICISSP_2027_Vorbereitung_2026-10-09/DOS_WORKFLOW_2026-10-09.md), dessen Kernbefunde hier eigenständig nachgeprüft wurden.
- vLLM v0.17.1: [Scheduler](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/core/sched/scheduler.py), [Konfiguration](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/scheduler.py), [Completion-Protokoll](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/entrypoints/openai/completion/protocol.py), [Metriklogger](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/metrics/loggers.py).
- Zusätzlich direkt im installierten Paket geprüft: `vllm/v1/engine/core.py`, `vllm/v1/utils.py`, `vllm/config/cache.py`, `vllm/config/observability.py`, `vllm/engine/arg_utils.py`, Serving/Input-Prozessor, KV-Hashbildung und Blockpool.
- [Offizielle Prefix-Caching-Dokumentation](https://docs.vllm.ai/en/v0.17.1/design/prefix_caching/).
