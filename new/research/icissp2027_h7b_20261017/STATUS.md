# H7b-Arbeitsstand, 17.10.2026

1. Die Literaturprüfung und Hypothese H7b stehen in `LITERATURPRUEFUNG_UND_ABGELEITETE_HYPOTHESE_20261017.md` im zugehörigen Codex-Arbeitsverzeichnis. Diese Umsetzung fügt den fehlenden **Offline-Evaluator** hinzu; sie ist noch kein Experiment.
2. Der Evaluator prüft SSE-Vollständigkeit, D8-Dispatch, zeitliche Kausalität, 496/0-Cache-Hits als getrennte Manipulationsprüfung, zeitlich getrennte Kalibrierung, gelernte Baselines, Shadow-KDE und eine auf Vorauswahl und Modell angewandte Permutationskontrolle. Der breite-Tail-KDE-Gegenfall wird numerisch korrekt integriert.
3. **26 Offline-Tests bestanden.** Ein End-to-End-CLI-Lauf verarbeitet 128 ausdrücklich synthetische Trials (32 Kalibrierung, 96 Test). Er meldet `synthetic_plumbing_only`, `hypothesis_demonstrated=false`, `submission_ready=false`.
4. Alte Serving-Daten wurden lediglich für eine öffentliche Timing-Projektion gelesen. Sie enthalten weder randomisierte H7b-Seeds noch einen H7b-Query und belegen keinen Effekt.
5. Nächster Implementierungsschritt: einen **neuen** H7b-Collector an den vorhandenen Executor und SSE-Callback anschließen, mit frischen V-Seeds, randomisiertem geheimen Bit, tatsächlichem Q512/1-Dispatch an D8 und vollständiger Rohdatenspeicherung. Historische Runner bleiben unverändert. Danach gegen denselben Eingabevertrag offline adversarial testen.
6. Lokale Live-Ausführung ist derzeit durch fehlende Kommunikation von `nvidia-smi` mit dem NVIDIA-Treiber blockiert. Eine Treiberreparatur oder ein GPU-Lauf wurde nicht vorgenommen.

**Entscheidung:** GO für Collector und begrenzten Machbarkeitspiloten nach technischem Preflight; NO-GO für einen empirischen H7b-Befund oder eine Einreichungsbehauptung jetzt. Eine spätere Überprüfung der tatsächlichen PrefixWall-Flag-Zyklen ist erforderlich, falls aus dem Kriteriumstest ein Implementierungs-Claim werden soll.
