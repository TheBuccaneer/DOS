# Einbau in DOS

Stand: 10. Oktober 2026. Dieses Paket enthält implementierte technische Werkzeuge, keine neue Lastkampagne und keinen bestätigten GPU-Gate-Pass.

## Zielorte

Den Inhalt dieses Paketordners mit dem bestehenden DOS-Root zusammenführen:

| Inhalt des Pakets | Ziel im Projekt |
| --- | --- |
| `new/scripts/cache_isolation_gate/` | `/home/rock/projects/DOS/new/scripts/cache_isolation_gate/` |
| `new/docs/icissp_2027_gate/` | `/home/rock/projects/DOS/new/docs/icissp_2027_gate/` |
| `EINBAU.md`, `HANDOVER.md`, `PACKAGE_STATUS.json`, `SHA256SUMS.txt` | Als Paketinformationen behalten; sie müssen nicht ins DOS-Root kopiert werden. |

Die ZIP enthält einen äußeren Ordner `DOS_ICISSP_Technikpaket_v2`. Dessen Unterordner `new` mit DOS/new zusammenführen. Nicht die alten Skriptordner ersetzen. Falls `cache_isolation_gate` bereits existiert, zuerst dessen Änderungen vergleichen; dieses Paket nicht blind darüberkopieren.

Es gibt keine automatischen Installations-, Git-, Commit- oder Push-Schritte. Der vorhandene Checkout wurde bei der Erstellung nicht verändert.

## Erster ausführbarer Schritt

Nach dem Einbau im DOS-Root:

```bash
bash new/scripts/cache_isolation_gate/preflight.sh /home/rock/projects/DOS /home/rock/projects/porto/.venv-py313-repair/bin/python3
```

Das prüft Quellen, Umgebung, vorhandene historische Daten und die neuen Offline-Tests. Es startet keinen Server. Die JSON-Berichte landen in einem ausdrücklich ausgegebenen temporären Evidenzordner. Für dauerhafte Nachvollziehbarkeit diesen Ordner anschließend unter `new/runs/cache_isolation_gate/evidence/` sichern. Exit-Code 0 bestätigt die Offline-Prüfung, nicht GPU-Ladbarkeit oder Studiengültigkeit.

## Serverstart mit der bisherigen Vorlage

Vorhandene vLLM-Umgebung aktivieren. Beispielsweise:

```bash
source /home/rock/projects/porto/.venv-py313-repair/bin/activate
```

Dann für eine Funktionsprüfung:

```bash
MAX_NUM_BATCHED_TOKENS=1024 bash new/scripts/cache_isolation_gate/run_server.sh llama 0
```

1024 ist hier ein Startbeispiel, keine empirisch bestätigte Studienbudgetstufe. Der Starter ist direkt aus `new/scripts/chunk_budget_screen/run_server.sh` abgeleitet. Er nutzt weiterhin `vllm serve`, dasselbe Llama-Modell, Seed 20260711, max_num_seqs 16, max_model_len 8192, GPU-Auslastungsparameter 0.90 und Tensorparallelität 1.

Die wissenschaftlich notwendigen Änderungen sind ausdrücklich dokumentiert: Prefix-Caching an, Iterationsdetails an, Weight-Offload zwingend null, partielle Prefill-Parameter explizit konstant. Dazu kommen ein eigener Logordner, Beschränkung auf 127.0.0.1 und Versionsprüfung auf vLLM 0.17.1. Die originale Vorlage und ein vollständiger Diff stehen im Unterordner `reference/`.

Der Starter übernimmt das bekannte Testschlüsselverhalten der Vorlage. Für die Clientprobe muss `VLLM_API_KEY` mit dem tatsächlich verwendeten Serverschlüssel übereinstimmen; Schlüsselwerte gehören nicht in Manifeste oder Chatübergaben.

Optional akzeptiert der Starter `KV_CACHE_MEMORY_BYTES` als explizite Speicherzuweisung. Das Paket legt keinen Grenzwert fest und kalibriert keine Kapazitätserschöpfung. Ohne diesen Wert arbeitet vLLM mit automatischer Kapazität. Weder ein gleicher angeforderter Bytewert noch gleiche GPU-Auslastung beweisen gleiche effektive Blockzahl. Den Laufzeitnachweis separat erfassen.

Originale historische Runner erwarten andere Start-/Hashverträge. Sie deshalb nicht einfach mit diesem Starter als angeblich unveränderte offizielle Kampagne aufrufen. Die neue Funktionsprüfung ist getrennt gekennzeichnet.

## Was als Nächstes ausführbar ist

Siehe `new/docs/icissp_2027_gate/WORKFLOW.md`: lokaler Metrikrecorder, begrenzte sequenzielle Cache-Probe mit dem vorhandenen Raw-SSE-Executor, Iterationslogparser und Offline-Auswertung. Siehe `MESSVERTRAG.md` für Datenfelder und `INTEGRATION.md` für die Anschlussstellen des vorhandenen Samplers.

Das vollständige Acht-Zellen-Experiment bleibt bis zum Runtime-Gate offen. Ein automatischer Treiber zum gezielten Erzeugen von KV-Erschöpfung oder Verfügbarkeitsstörungen ist nicht enthalten.
