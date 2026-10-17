# DOS / Cache-Isolation: technische Werkzeuge v2

Einbau und erster Befehl: `DOS_ICISSP_Technikpaket_v2/EINBAU.md`.
Dokumentation nach Einbau: `new/docs/icissp_2027_gate/`.

| Datei | Implementierte Funktion |
| --- | --- |
| `run_server.sh` | Aus dem bisherigen Budget-Starter abgeleiteter vLLM-0.17.1-Start; Prefix-Caching und Iterationsdetails an, Offload null. |
| `verify_sources.py` | SHA256-Abgleich der wiederverwendeten Originalkomponenten. |
| `inspect_environment.py` | Vorhandene Umgebung/GPU-Metadaten lesen, kein Modellstart. |
| `preflight.sh` | Quellen, Umgebung, historische Daten und Regressionstests. |
| `audit_existing.py`, `trace_contract.py` | Korrigierte historische Rohdaten-/Paarprüfung mit fortgesetzter Fehlerberichterstattung. |
| `cache_smoke.py` | Drei kleine sequenzielle Cache-Requests über den vorhandenen Raw-SSE-Executor. |
| `cache_eviction_control.py` | Sechs strikt sequenzielle Requests A1–A2–B1–B2–B3–A3; je Aufruf genau eine Kapazität, append-only JSONL mit vollständigen SSE-Records und Cache-Zählerdifferenzen. |
| `overlap_pilot.py`, `overlap_contract.py` | Einzelne D/B-Episode mit Token-8-Trigger, dauerhaftem SSE-/Metrikjournal und getrennten technischen Gates; reine Offline-Prüfung mehrerer Episoden über `audit`. Siehe [OVERLAP_PILOT.md](OVERLAP_PILOT.md). |
| `record_metrics.py` | Fortlaufende begrenzte lokale /metrics-Aufzeichnung, keine Completion-Requests. |
| `metrics_contract.py` | Abschlussadapter für vollständige Samples eines vorhandenen Samplers. |
| `summarize_metrics.py` | Labeltreue Serien, fehlende Werte und Resets berücksichtigen. |
| `parse_iteration_logs.py`, `compare_budget_logs.py` | Tatsächliche Context-/Generation-Tokens pro Iteration auslesen und vergleichen. |
| `validate_contract.py` | Vollständigkeit des Studienvertrags und acht Faktorbezeichnungen prüfen. |
| `extract_primary_endpoint.py` | Feste Kohorte, späterer Tokenindex, gemeinsame Beobachtungszeit. |
| `build_endpoint_table.py` | Bestehende Episoden und tatsächliche Konfigurationsmetadaten verbinden. |
| `analyze_pairs.py` | Gepaarte H-, E- und Gamma-Werte; Bootstrap vollständiger Replikatblöcke. |
| `contracts/` | Auszufüllende Studien-, Runtime-, Konfigurations- und Episode-Map-Vorlagen. |
| `tests/` | Synthetische Fixtures; keine wissenschaftlichen Messdaten. |
| `reference/` | Unveränderte Startervorlage, Quellenangabe und vollständiger Delta-Patch. |

Offline-Aufrufe funktionieren mit Python-Standardbibliothek. Server und Cache-Funktionsprobe verwenden die bereits vorhandene vLLM/httpx-Umgebung. Die Funktionsprobe prüft den Originalexecutor-Hash; nicht bei unbekanntem Code automatisch fortfahren.

Exit-Code 0 bedeutet den Erfolg des jeweiligen technischen Aufrufs. Kein Programm dieses Pakets bestätigt automatisch das wissenschaftliche Runtime-Gate. Die Template-Dateien enthalten absichtlich Nullfelder; sie sind nicht als ausführbare Studienkonfiguration ausgegeben.

Nicht enthalten: autonomer KV-Druck-/Interferenztreiber, vollständige Acht-Zellen-Livekampagne, automatische Integration in historische Runner, Mechanismusbeweis oder Manuskript.

## Sequenzieller Eviction-Kontrollversuch (noch kein GPU-Lauf)

Der Starter akzeptiert optional `MAX_MODEL_LEN` (Standard 8192) und
`NUM_GPU_BLOCKS_OVERRIDE` (ohne Vorgabe weiterhin automatische Kapazität).
Beide Werte werden geprüft; `KV_CACHE_MEMORY_BYTES` und der Block-Override
dürfen nicht gleichzeitig gesetzt sein. Das Startup-Log enthält die
angeforderten Werte. Der Kontrollrunner verlangt zusätzlich die vLLM-Zeilen
`Chunked prefill is enabled with max_num_batched_tokens=...`,
`Overriding num_gpu_blocks=...`, `GPU KV cache size: ... tokens` und
`Maximum concurrency for ... tokens per request`. Er prüft die tatsächlich
gemeldete Blockzahl und Blockgröße in `cache_config_info` vor jedem Request.
Die APIServer-PID und UTC-Startzeit aus dem Log müssen zur laufenden
Linux-Prozessinstanz und deren `/metrics`-Startzeit passen. Vor A1 müssen
die Anfrage- und Cache-Zähler null sein; jede Kapazitätsstufe benötigt
damit eine eigene frische Serverinstanz mit zugehörigem Log.

Offline-Kandidaten für zwei **getrennte**, frisch gestartete Serverläufe:
`MAX_MODEL_LEN=256`, `MAX_NUM_BATCHED_TOKENS=512`, CPU-Offload 0 und
`NUM_GPU_BLOCKS_OVERRIDE=24` bzw. `64`. Bei 16 Tokens pro Block bleibt nach
dem Null-Block Platz für 23 bzw. 63 Blöcke. Jeder 128-Token-Prompt mit zwei
Ausgabetokens braucht höchstens 9 aktive Blöcke. A2 kann wegen der
Neuberechnung des letzten Promptblocks eine zusätzliche Hash-Kopie anlegen;
die drei unterschiedlichen B-Prompts benötigen bis zu 27 weitere Blöcke.
Damit ist Verdrängung bei 24 plausibel und Erhaltung bei 64 plausibel. Das
ist eine Codeableitung, keine bestätigte Laufzeitkonfiguration. Tatsächliche
Blockzahl, Blockgröße, Logs, Trefferdifferenzen und Prefill-Arbeit sind Gates.

Der Runner nimmt die vorhandene **öffentliche** 128-ID-Datei für A. Aus den
ersten zwei unterschiedlichen IDs erzeugt er deterministisch drei andere
erste 16er-Blöcke für B; die übrigen 112 IDs bleiben gleich. Er speichert
Datei-SHA256, alle sechs Tokenlisten und deren kanonische SHA256 im JSONL.
Je Request enthält es den vollständigen Raw-SSE-Record, rohe Metrik-Samples,
Zählerdifferenzen und, wenn eindeutig, die zugehörigen Iterationslog-Tokens.
Jede Zeile wird sofort geschrieben und synchronisiert; ein Fehler beendet
die Folge und erhält die bisherigen Daten. Ein vorhandener Ergebnisname
wird nie überschrieben. Verzögerte oder widersprüchliche Metriken machen den
Lauf ungültig. A1 muss 0, A2 genau 112 und B1–B3 jeweils 0 Cache-Hits
haben; sonst bricht die Kontrolle vor A3 ab. `valid_control_run` bedeutet
nur vollständige, zuordenbare
Beobachtung einer Kapazitätsstufe, keinen H1-Nachweis.

Nach ausdrücklicher Freigabe wäre der Aufruf pro Kapazitätsstufe:

```bash
MAX_MODEL_LEN=256 NUM_GPU_BLOCKS_OVERRIDE=24 MAX_NUM_BATCHED_TOKENS=512 bash new/scripts/cache_isolation_gate/run_server.sh llama 0
# In einer separaten Shell, mit dem zu diesem Server gehörenden Startup-Log:
python3 -B new/scripts/cache_isolation_gate/cache_eviction_control.py \
  --root /home/rock/projects/DOS \
  --prompt-token-ids new/runs/cache_isolation_gate/public_prompt_token_ids_128.json \
  --server-log PFAD_ZUM_AKTUELLEN_SERVERLOG \
  --expected-num-gpu-blocks 24 --expected-max-model-len 256 \
  --expected-budget 512 \
  --output new/runs/cache_isolation_gate/eviction_24_EINDEUTIGER_NAME.jsonl
```

Für den zweiten Lauf `24` durch `64` ersetzen und einen neuen Ergebnisnamen
verwenden. Die zwei Aufrufe sind eine Anleitung, keine hier ausgeführte Messung.

## Minimaler Overlap-Pilot

Der vollständige technische Ablauf, die Gate-Definitionen, Grenzen und
Offline-Testbefehle stehen in [OVERLAP_PILOT.md](OVERLAP_PILOT.md).
Der Starter aktiviert jetzt `--enable-prompt-tokens-details`, damit sich
Cache-Treffer von D und B einzeln zuordnen lassen. Bestehende Serverlogs
werden bei einer Namenskollision abgewiesen und erhalten.
