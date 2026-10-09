# DOS – erster Workflow und technischer Gate-Check

Stand: 9. Oktober 2026. Übergabe aktualisiert: Alle neuen Dateien liegen gemeinsam im übertragbaren Vorbereitungsordner. Arbeitsgrundlage: `/home/rock/Desktop/Arbeitsanweisung.txt` und der freigegebene Ordner `/home/rock/projects/DOS/`. Die Arbeitsanweisung dient als fachliche Vorgabe für diese Bestandsaufnahme. Der konkrete Auftrag in der Unterhaltung ist: Projekt kennenlernen, einen Workflow anlegen, fehlende Informationen besprechen und die Experimentvorbereitung beginnen.

## Entscheidung nach der ersten Sichtung

**Die Messbasis ist wiederverwendbar; das neue Acht-Zellen-Experiment ist noch nicht technisch bestätigt.** Für die Vorbereitung ist genug Material vorhanden. Für einen belastbaren Start fehlen die aktuelle Umgebung des separaten GPU-Rechners, wirksame Cache-/Budgetnachweise und eine vollständige Metrikaufzeichnung. Der Nutzer hat bestätigt, dass die RTX 3090 auf einem anderen Rechner steht.

Angelegt wurden ein reproduzierbarer Offline-Audit und eine ausschließlich lesende Umgebungsprüfung. Die bisherigen Messskripte und Bestandsdaten bleiben die Referenz. Die neue Untersuchung muss als eigene Studie geführt werden; historische Resultate werden nicht als Resultate der Cache-Isolationsfrage ausgegeben.

## 1. Projektstand und belastbare Evidenz

Der geprüfte lokale Stand ist Commit `d7b2bced85fb54459362f9eb5b65f8333ff24bd1` vom 6. Oktober 2026. Vor der Vorbereitung war der Git-Arbeitsstand sauber. `new/` ist der vorhandene Ordnername; ein separates `New/` wurde nicht gefunden.

Die Root-Berichte `DOS_Bestandsdaten_Auswertung_2026-10-06.md` und `DOS_Spaete_Requests_Kritikbewertung_2026-10-06.md` enthalten die jüngste gefundene Auswertung. `new/README.md` beschreibt noch die ältere Offload-Profilmatrix; es bildet den aktuellen Forschungsauftrag nicht vollständig ab. `paper/` enthält nur `.gitkeep`. Manuskript, Originalreviews und der genannte Literaturabgleich wurden in Root, `new/` und `docs/` nicht gefunden. Das lokale ChatGPT-Projektverzeichnis `sources/` enthält derzeit keine Dateien.

Die folgende Auswahl wurde neu offline geprüft, ohne Messrequests zu senden:

| Historische Kampagne unter `new/runs/` | Episoden | Paare | Requests | Ausgabetokens |
|---|---:|---:|---:|---:|
| `chunk_budget_screen/results/official_20260716T075455Z` | 36 | 18 | 792 | 47.232 |
| `server_waiting_confirmation/llama/official_campaign_20260804T141510Z` | 32 | 16 | 704 | 41.984 |
| `server_waiting_confirmation/qwen/official_campaign_fixed_20260803T183032Z` | 32 | 16 | 704 | 41.984 |
| Summe | 100 | 50 | 2.200 | 131.200 |

Geprüft wurden die Rekonstruktion der Token-IDs aus Raw-SSE, Tokenzahlen, monotone Empfangszeiten, erster/letzter Tokenzeitpunkt, Request-Zeitgrenzen und die Identität der Decode-Requests innerhalb der Paare. Dabei wurden keine Fehler gefunden. Je Kampagne ist die Paarreihenfolge exakt ausbalanciert. Diese begrenzte Prüfung ersetzt weder den vorhandenen vollständigen Auditor noch die Prüfung des neuen Experiments.

Die drei historischen Manifeste nennen vLLM **0.17.1**, Torch **2.10.0** und Python **3.13.7**. Historische Python-Pfade sind `/home/rock/projects/porto/.venv/bin/python3` beziehungsweise `/home/rock/projects/porto/.venv-py313-repair/bin/python3`; sie sind Hinweise für den anderen Rechner, keine hier bestätigte Installation. Das Budget-Manifest nennt eine RTX 3090 mit 24.576 MiB und Treiber 580.159.03. Der aktuelle Stand des Messrechners bleibt offen.

Der SHA-256 des geschützten Prefill-Basisrunners ist weiterhin `981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75`. Auch der geprüfte Budget-Runner entspricht seinem historischen Manifest. Weitere Hashes und die Prüfergebnisse stehen in `bestand_audit.json` im selben Vorbereitungsordner.

## 2. Wiederverwendbare Komponenten

| Komponente | Verwendung | Einschränkung für die neue Studie |
|---|---|---|
| `new/scripts/prefill_confirmation/run_prefill_confirmation.py` | Streaming, Raw-SSE, Token-IDs, monotone Zeitstempel, Seed-/Prompt-Identitäten | Eigener historischer Vertrag; keine tenantweisen Salts im vorhandenen Request-Payload |
| `new/scripts/chunk_budget_screen/` | Explizites Budget, gepaarte Reihenfolge, Provenienz | Prefix-Caching abgeschaltet; keine Isolations- oder KV-Kapazitätsachse |
| `new/scripts/server_waiting_confirmation/` und `_llama/` | Transportzeitpunkte, Kohortenprüfung, separater Metriktransport | Dynamisch ausgewählte Kohorten sind nicht automatisch zwischen Paaren identisch |
| `audit_server_waiting_diagnostic.py` | Vorhandener strenger Raw-Trace-/Provenienz-Auditor | Abnahme gilt für sein historisches Schema, nicht automatisch für ein erweitertes Schema |
| `new/scripts/prefill_confirmation/audit_prefill_server_logs.sh` | Sichtung von Preemption-/KV-/Fehlerhinweisen | Kein Logtreffer beweist keine ausreichende KV-Reserve |
| Root-Auswertungen und `new/scripts/workshop_prefill_analysis/` | Datenverständnis und Auswertungsmuster | Bisherige Endpunkte/Claims nicht unverändert übernehmen |

Die Vorbereitung benötigt keine neue allgemeine Messplattform. Erweiterungen sollten klein, separat versioniert und an den vorhandenen Datenverträgen geprüft werden.

## 3. Cache- und Budgetparameter: was bestätigt ist

**Repository-Befund:** Die geprüften relevanten Serverstarter verwenden `--no-enable-prefix-caching`. Der Budget-Starter übergibt `--max-num-batched-tokens` ausdrücklich. Die alten Daten zeigen verschiedene clientseitige Pausen, aber enthalten keinen vollständigen internen Scheduling-Verlauf, der den neuen Faktor abnimmt.

**Versionsbezogene Prüfung:** Im V1-Scheduler von vLLM 0.17.1 stammen die pro Iteration verfügbaren Tokens aus `max_num_scheduled_tokens`. Positive `long_prefill_token_threshold`-Werte begrenzen neue Tokens pro Request; das ist konkreter als die bloße CLI-Beschreibung „langes Prefill“. Die Namen `max_num_partial_prefills` und `max_long_partial_prefills` werden in der geprüften Standard-Scheduler-Datei nicht verwendet. Ihre Existenz in der Konfiguration bestätigt daher keine Laufzeitwirkung. Für eine abweichende oder asynchrone Scheduler-Klasse muss deren tatsächlicher Code geprüft werden. [Versionierter Scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/v1/core/sched/scheduler.py).

`max_num_batched_tokens` bezeichnet ein Gesamtbudget, kein ausschließliches Prefill-Budget. Die Konfiguration erläutert außerdem, dass `max_num_scheduled_tokens` normalerweise daraus abgeleitet wird, in Sonderkonfigurationen aber abweichen kann. Deshalb beide effektiven Werte erfassen; die Partial-Prefill-Parameter zunächst konstant halten. [Versionierte Scheduler-Konfiguration](https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/config/scheduler.py).

Prefix-Caching und `cache_salt` sind für diese Version dokumentiert. Salts trennen die Wiederverwendung von KV-Blöcken; sie reservieren keine getrennten Speicherbudgets für Tenants. Ein stabiler Salt pro Tenant erhält Wiederverwendung innerhalb des Tenants. Ein neuer Salt pro Request wäre ein anderer Faktor. [Prefix-Caching und Isolation](https://docs.vllm.ai/en/v0.17.1/design/prefix_caching/). Der Completion-Request validiert einen übergebenen Salt als nichtleere Zeichenkette. [Completion-Protokoll](https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/entrypoints/openai/completion/protocol.py).

**Offene Abnahme:** Diese Quellenprüfung gilt für den offiziellen Versionsstand. Erst die installierte Version, tatsächliche Scheduler-Klasse, effektive Konfiguration und ein begrenzter Funktionstest auf dem Messrechner bestätigen die lokale Wirkung. Bis dahin sind Cache-Policy und Budget keine abgenommenen Experimentalparameter.

## 4. Fehlende Messpunkte und Designkorrekturen

1. **Verlauf nach Beginn der Intervention fehlt.** Im Server-WAITING-Runner wird `list(metrics_sampler.samples)` beim Trigger in `trigger.metrics_raw_samples` kopiert. Der Sampler läuft danach weiter, seine späteren Samples werden dort jedoch nicht nachgetragen. In den 64 geprüften Episoden gibt es 21.647 gespeicherte Samples und **kein einziges nach dem logischen Trigger**. Die Funktion beendet den Sampler nach den Requests; das ist ein guter Ort für eine künftige separate vollständige Aufzeichnung.
2. **Kohortenidentität fehlt für den neuen Paarendpunkt.** Die dynamisch ausgewählten aktiven Request-Mengen unterscheiden sich in allen 32 Server-WAITING-Paaren. Die Promptpopulation innerhalb jedes Paars ist dennoch identisch. Für die neue Frage eine Decode-Kohorte vorab anhand fester Request-Identitäten bestimmen und vor Intervention bestätigen; später nicht nach Abschlussrang oder Latenz neu auswählen.
3. **Cache-Regel und Cache-Zustand fehlen im Datenvertrag.** Tenantzuordnung, Policy, reproduzierbare öffentliche Präfixidentität und kontrollierter Ausgangszustand müssen im Laufmanifest nachvollziehbar sein. Gleich angebotene Token-Arbeit ist nicht gleich ausgeführte Prefill-Arbeit; diese Differenz ist zu messen.
4. **Tatsächliche KV-Kapazität fehlt als kontrollierter Faktor.** Gleiche `gpu_memory_utilization` allein garantiert bei verschiedenen Budgets keine identische nutzbare KV-Kapazität. Effektive Blockzahl, Blockgröße und KV-Datentyp müssen innerhalb der Policy-/Budgetvergleiche übereinstimmen. Keine konkrete Kapazitätsgrenze aus den Altlogs behaupten.
5. **Der neue primäre Endpunkt fehlt.** Die Raw-SSE-Daten ermöglichen den Empfangszeitpunkt eines späteren Tokenindex. Dafür Index, Zeitursprung und Kohortenaggregation vorab festlegen. Mehrtoken-SSE-Events erlauben nur einen gemeinsamen Empfangszeitpunkt, keine erfundenen individuellen Generierungszeiten.

Bereits gespeicherte Rohmetriken enthalten KV-Belegung, Prefix-Queries/-Hits, Preemptions, Prompt-Tokens nach Quelle und Iterationshistogramme. Diese Metriken können voraussichtlich ohne Engine-Umbau weiterverwendet werden. Der aktuelle Parser reduziert sie jedoch auf Running/Waiting. Eine vollständige Zeitreihe und passende Zählerdifferenzen müssen künftig erhalten bleiben. Iterationshistogramme trennen Prefill und Decode nicht pro Schritt; für einen strengen Scheduling-Nachweis kann deshalb noch ein kleiner Beobachtungspunkt fehlen. `prompt_tokens_recomputed` bezeichnet in dieser Version insbesondere wiederberechnete gecachte Tokens für den Forward-Pass und ist kein hinreichender Preemption-Recompute-Nachweis. [Versionierter Metriklogger](https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/v1/metrics/loggers.py).

## 5. Workflow mit klaren Abnahmeschritten

### Schritt A – Bestandsaufnahme abschließen

- [x] Root und `new/` zuordnen; jüngste Berichte sichten.
- [x] Drei relevante historische Kampagnen und ihre Manifeste prüfen.
- [x] Wiederverwendbare Messkomponenten und konkrete Lücken dokumentieren.
- [x] Versionsbezogene Cache-/Scheduler-Semantik gegen Primärquellen prüfen.
- [ ] Später gemeinsam am Messrechner den dortigen Repository-Pfad, die bisherige Python-Umgebung und das vorhandene Modell zuordnen.
- [ ] Manuskript, Originalreviews und vorhandenen Literaturabgleich lokalisieren.

### Schritt B – Offline-Vorbereitung und Umgebungsnachweis

Die beiden mitgelieferten Programme arbeiten ausschließlich lesend und geben JSON auf stdout aus. Der Offline-Audit wurde hier ausgeführt. Die Umgebungsprüfung wurde hier ebenfalls ausgeführt und bestätigt nur, dass diese aktuelle Arbeitsumgebung die Messabhängigkeiten nicht bereitstellt. Sie muss zusätzlich mit der bisherigen Python-Umgebung auf dem GPU-Rechner ausgeführt werden; README und Aufrufvorlagen liegen im Vorbereitungsordner.

Abnahme: Versionen, GPU, Quellcodehashes und lokal vorhandenes Modell eindeutig dokumentiert. Modell-/Tokenizerrevision und effektive Scheduler-Klasse anschließend ergänzen. Noch kein Ready-Urteil aufgrund installierter Pakete allein.

### Schritt C – minimale Anpassungen für einen kontrollierten Funktionsversuch

Die notwendigen Änderungen sind als Arbeitsliste vorbereitet, noch nicht im historischen Runner umgesetzt:

| Änderung | Abnahme |
|---|---|
| Cache-Policy und Tenantidentität im neuen Studienschema | Policy wirkt tatsächlich; Wiederverwendung innerhalb/zwischen Tenants entspricht dem Design |
| Vollständige zeitgestempelte Metrikaufzeichnung | Vor, während und nach dem begrenzten Test liegen Samples mit Qualitätskennzeichnung vor |
| Effektive Serverkonfiguration und KV-Kapazität im Manifest | Innerhalb jedes Vergleichs ist die nutzbare Kapazität konstant |
| Vorab feste Decode-Kohorte und Tokenindex-Endpunkt | Identitäten, Zeitursprung und Aggregation werden ohne Ergebniskenntnis festgelegt |
| Schema-/Auditor-Erweiterung | Fehlende Salts/Policy, falsche Kohorten, lückenhafte Zeitreihen und nicht wirksame Budgets werden erkannt |

Erst ein begrenzter Funktionstest im eigenen isolierten Testsystem mit vorhandenen öffentlichen Testdaten. Zunächst die Cache-Funktion und die Beobachtbarkeit einzeln prüfen. Keine Parameterwerte zum Erzeugen von Überlastung aus den Altversuchen übernehmen. Der Test soll die Messkette und Schutzkonfiguration bestätigen.

### Schritt D – technisches Gate für die Acht-Zellen-Studie

Die acht Zellen stehen als **Entwurf** in `diagnose_design.json`: Shared/Isolated × Small/Large × ausreichende/begrenzte KV-Kapazität. Zahlenwerte und ein ausführbarer Messrunner werden erst nach den vorigen Nachweisen festgelegt. Das Design ist keine startbare Kampagne.

Gate bestanden nur, wenn:

- die Cache-Regel tatsächlich umgesetzt ist;
- die beiden Budgeteinstellungen unterschiedliche beobachtbare Scheduling-/Prefill-Verläufe erzeugen;
- die festgelegte KV-Kapazität innerhalb der Vergleiche gleich bleibt;
- dieselbe Decode-Kohorte und dieselbe angebotene Arbeit verfolgt werden;
- Primärendpunkt und vollständige Diagnosedaten auswertbar sind.

Fehlt ein Nachweis, wird zuerst die Ursache geklärt. Weitere Faktorstufen ersetzen keinen unwirksamen Faktor.

## 6. Auswertung vorab festlegen

Vorschlag für den primären Endpunkt: Zeit vom gemeinsamen, vorab definierten Beobachtungsbeginn bis **alle** festgelegten Kohortenmitglieder den späteren Tokenindex erreicht haben. Innerhalb eines Paars ist H die Zeit mit Zusatzlast minus Referenzzeit. Das entspricht einer Kohorten-Zielzeit; ein Median einzelner Requestzeiten wäre ein anderer Endpunkt und sollte gegebenenfalls sekundär bleiben.

Tokenindex, Beobachtungsbeginn, tolerierbare Ausfälle und praktisch relevante Mindestwirkung bleiben vor dem ersten neuen Ergebnis zu bestätigen. Nicht erreichte Tokenindizes als Ausfälle/Zensierung behandeln, nicht aus der Auswertung verschwinden lassen. Clientempfang ist kein serverinterner Ausführungszeitpunkt; lokale Admission-/Dispatchzeit und serverseitige Diagnosen separat berichten.

Für jede KV-Stufe separat: `E_p = H_p(small) − H_p(large)` und `Gamma = E_isolated − E_shared`. Positive Gamma-Werte allein belegen keine Umkehr; dafür müssten zusätzlich `E_shared < 0` und `E_isolated > 0` belastbar sein. Die Paar-/Blockstruktur in der Unsicherheitsrechnung erhalten. Requests und Tokens sind keine unabhängigen Wiederholungen. Anzahl der Wiederholungen anhand gewünschter Präzision und einer begrenzten Pilotstreuung festlegen, nicht nach dem Vorzeichen nachträglich erhöhen.

Die Arbeitsanweisung nennt H0, H1 und Messartefakte als konkurrierende Erklärungen. Der Workflow übernimmt keine davon als Ergebnis. Der FairCache-Volltextabgleich bleibt offen; ein Novelty-Urteil und Manuskriptarbeit folgen erst nach dem technischen Gate und einer belegten Literaturabgrenzung.

## 7. Rücksprache und nächste konkrete Aktion

Für die lokale Vorbereitung fehlt nichts Grundsätzliches. Alle neu erstellten Unterlagen und Werkzeuge sind im Ordner `ICISSP_2027_Vorbereitung_2026-10-09` gebündelt. Dieser Rechner ist ausdrücklich nur der Vorbereitungsrechner. Ein Remote-Zugang wird für die Vorbereitung nicht benötigt. Die tatsächlichen Pfade und die Python-Umgebung prüfen wir später gemeinsam direkt am Messrechner. Die historischen `porto`-Pfade erleichtern dort die Suche. Die Fundorte von Manuskript, Reviews und Literaturabgleich können dann ebenfalls ergänzt werden.

Nächste Aktion: Den gesamten Vorbereitungsordner auf den Messrechner mitnehmen. Sobald der Nutzer dort ist, gemeinsam die vorhandene Umgebung prüfen und mit den historischen Manifesten abgleichen. Danach die kleine Schema-/Observability-Erweiterung konkretisieren und den Funktionsnachweis abnehmen. Bis dahin werden hier keine GPU-Messungen und keine Experimentkampagne gestartet. Ein Manuskript wurde nicht begonnen.

Die Termine 10., 13., 16., 19., 21. und 22. Oktober stammen aus der Arbeitsanweisung und dienen als interne Kontrollpunkte. Einreichungsdeadline und Formatvorgaben wurden in dieser technischen Bestandsaufnahme nicht extern bestätigt.
