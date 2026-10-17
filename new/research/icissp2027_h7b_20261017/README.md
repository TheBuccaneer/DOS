# ICISSP 2027 H7b: reproduzierbare Offline-Auswertung

Stand: 17.10.2026. Dieses Verzeichnis enthält die **Auswertung und ihren Eingabevertrag** für einen begrenzten H7b-Machbarkeitspiloten. Es enthält keinen H7b-Serving-Runner und keine neuen GPU-Messungen. Die bestehende Overlap-Infrastruktur bleibt unverändert.

## Fragestellung und Literaturgrenze

[PrefixWall v2](https://arxiv.org/html/2603.10726v2) bewertet die Überlappung von Hit-/Miss-TTFTs in einem gleitenden Fenster und kann selektive Isolation bei hoher Überlappung deaktivieren (§3.3). [Fill and Squeeze v2](https://arxiv.org/abs/2602.07878v2) nutzt eigene Stream-Latenzen als Indikator des Betriebszustands (§4, Anhang E). [InputSnatch v2](https://arxiv.org/html/2411.18191v2) zeigt bereits gelernte Cache-Timing-Klassifikatoren (§4.2–4.4). H7b prüft den **zusätzlichen Wert der vor dem Query beobachteten eigenen D-Token-Intervalle** gegenüber kalibriertem TTFT und dessen eigener zeitlicher Historie, während eine offen ausgewiesene Nachbildung der PrefixWall-Fensterregel „OFF“ meldet. Ein positives Ergebnis wäre zunächst eine Grenze dieser Entscheidungsvorschrift unter den getesteten Bedingungen; es wäre kein Nachweis eines realen PrefixWall-Bypasses.

Der primäre vorab gewählte Shadow-Punkt ist **W=32, θ=0,90**. W=16 und θ∈{0,50; 0,70; 0,90; 0,95; 0,99} sind Sensitivitätsansichten, keine nachträglich ausgewählten Primärergebnisse. Die KDE-Bandbreite und die ITL-Auswahlschwelle kommen ausschließlich aus dem Kalibrierungsteil.

## Eingabevertrag

Eine JSONL-Zeile pro Trial, `schema: "dos-h7b-trial-v1"`:

- `trial_id`: eindeutiger Name; `time_block`: zusammenhängender Zeitblock, der nur zu *einem* Split gehört.
- `source_kind`: `serving` oder `synthetic`; synthetische Zeilen werden nur mit `--allow-synthetic` verarbeitet.
- `split`: `calibration` oder `test`; Kalibrierung nur im gemeinsamen Namespace, mindestens 16 Trials und zwei Blöcke; sie muss vor allen Tests abgeschlossen sein.
- `namespace`: `shared` oder `isolated`. Der Test enthält beide Kontrollen. Das geheim randomisierte Cache-Mitgliedschaftsbit steht ausschließlich unter `evaluator.secret_bit`.
- `query_dispatch_ns`, `trigger_receipt_ns`: monotone `perf_counter_ns`-Zeitstempel des tatsächlichen Q-Dispatchs und des empfangenen D-Tokens 8.
- `reference_events`, `query_events`: vollständige rohe SSE-Listen. Jedes Element hat `raw_data` und `receive_perf_counter_ns`. Der letzte Eintrag ist `[DONE]`; die Usage enthält `prompt_tokens`, `completion_tokens` und für Q ausdrücklich `prompt_tokens_details.cached_tokens`.

Die Auswertung verlangt D192 mit 32 Ausgabetokens und einzeln empfangenen Tokens 1–8, Q512 mit einem Ausgabetoken, Q-Dispatch ab D8 und vor D9, vollständige D-Erfassung bis mindestens Token 28 sowie tatsächliche zeitliche Überlappung. Das öffentlich beobachtbare Feature `reference_pre_ms` ist der Median der vier **vor dem Dispatch abgeschlossenen** Intervalle D4→5 bis D7→8. `ttft_ms` beginnt beim tatsächlichen Q-Dispatch. `past_ttft_median_ms` nutzt nur vor dem aktuellen Dispatch abgeschlossene eigene Q-Anfragen. Cache-Zähler und das Seed-Bit dienen ausschließlich als Evaluator-Prüfdaten; ihre Werte gelangen nicht in die Merkmalsmatrix.

Die beabsichtigte neue Erfassung randomisiert pro Trial S∈{0,1}, benutzt frische Kandidatenpräfixe und gleich große V-Seeds. Q besteht aus 496 Kandidat-Tokens plus 16 frischen Suffix-Tokens. Erwartete Q-Hits: 496 nur bei S=1 im gemeinsamen Namespace, sonst 0. Abweichungen werden als Manipulationsfehler gespeichert und **nicht** still zu Negativen umgedeutet. Der Seed- und Request-Collector fehlt noch; die alten Overlap-Episoden erfüllen diesen Vertrag nicht.

## Auswertung und Grenzen

Die drei festgelegten gelernten Modelle nutzen (1) TTFT, (2) TTFT plus eigene abgeschlossene TTFT-Historie und (3) diese beiden Merkmale plus vorab beobachtete D-Intervalle. Der 32-Trial-Kalibrierungsteil prüft nur, ob diese Auswertung technisch funktioniert. Er repliziert die stärkeren Trainingsbedingungen von InputSnatch nicht. `ttft_rank` bleibt als einfacher Diagnosewert erhalten und ist **nicht** die Konkurrenzbaseline. Ein deterministischer Kontrolllauf vertauscht D-Intervalle innerhalb von Testblock und Namespace, sowohl im gemeinsamen Modell als auch in der Vorauswahl. Die ursprünglichen Auswahlquoten pro Block bleiben erhalten. AUCs ohne ausreichende Positiv-/Negativfälle sind `null`; kleine Gruppen tragen keinen stabilen Effektbeleg.

Der Shadow-Aktivator nutzt nur vor Q-Dispatch **abgeschlossene** Q-TTFTs, die zugehörigen serverseitigen Hit-Klassen und eine aus Kalibrierung eingefrorene KDE-Bandbreite. Bei fehlendem vollen Fenster oder weniger als vier Beobachtungen pro Klasse ist sein Zustand unbekannt. Er bildet weder PrefixWalls OwnerID-/Flag-Logik noch deren spätere Reaktivierung ab. Für eine Aussage über reale Schutzwirkung wären Implementierung, Metadatenzustand und ON→OFF→ON→OFF-Zyklen gesondert zu prüfen.

Die Analyse meldet selbst bei Serving-Eingaben `hypothesis_demonstrated: false` und `submission_ready: false`. Punkt-AUCs, eine einzelne Shadow-Einstellung oder ein künstliches Beispiel dürfen nicht als Publikationsbefund gelesen werden. Ein belastbarer positiver Claim verlangt passende Serving-Trials, Manipulationsvalidität, ausreichende ausgewählte Fälle, Unsicherheitsschätzung und Vorteile gegenüber der zeitlichen Baseline, die unter Permutation und Salt-Trennung verschwinden. Ohne solche Fälle bleibt der Pilot unentschieden.

## Reproduktion

Aus diesem Verzeichnis mit Python 3.12 und den Versionen in `requirements.lock`:

```bash
python -m pip install -r requirements.lock
python -m unittest discover -s tools -p 'test_h7b.py' -v
python tools/h7b.py --input fixtures/synthetic_trials.jsonl --output output/synthetic_plumbing.json --allow-synthetic
```

`fixtures/synthetic_trials.jsonl` ist ausdrücklich ein **künstlicher Funktionstest**. Sein berechneter AUC-Wert ist kein Forschungsergebnis. `evidence/legacy_projection.json` liest aus einer alten Episode nur zwei öffentliche Zeitmerkmale; diese Episode wird weder trainiert noch als H7b-Trial umetikettiert. `reference/reanalyze.py` ist eine unveränderte Kopie des bisherigen unabhängigen SSE-Parsers; `evidence/provenance.json` hält dessen Hash fest. `dependency_install.json` dokumentiert den Installationslauf. Die lokale `.venv` ist bewusst nicht Teil des ZIPs.

## Nächster Versuchsentscheid

**GO** für einen getrennt zu prüfenden H7b-Collector auf Basis des bestehenden SSE-Executors und D8-Triggers; anschließend ein begrenzter randomisierter 128-Trial-Machbarkeitspilot (32 Kalibrierung, 64 shared, 32 isolated). **NO-GO** für einen Privacy- oder PrefixWall-Bypass-Claim auf Basis dieses Offline-Pakets. Der lokale NVIDIA-Treiber ist derzeit über `nvidia-smi` nicht erreichbar. Es wurden weder Server gestartet noch GPU-Anfragen ausgeführt.
