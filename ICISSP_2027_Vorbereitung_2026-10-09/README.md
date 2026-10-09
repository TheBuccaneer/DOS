# Vorbereitung Cache-Isolation – 9. Oktober 2026

Alle neu erstellten Dateien liegen in diesem Ordner. Der zentrale Arbeitsstand steht in `DOS_WORKFLOW_2026-10-09.md` direkt daneben. Einstieg: `HIER_STARTEN.md`.

## Dateien

- `audit_existing.py`: lesende Prüfung der drei ausdrücklich ausgewählten historischen Kampagnen; keine Imports von Runnern, keine Messrequests.
- `bestand_audit.json`: Ergebnis dieser Prüfung, einschließlich Quellcodehashes und gemessener Datenlücken.
- `inspect_environment.py`: lesende Paket-, GPU- und Quellcodeinventur für den separaten Messrechner; kein Modellladen, kein Download, kein Dump von Umgebungsvariablen oder Zugangsdaten.
- `vorbereitungsrechner_environment.json`: Inventur der aktuellen Arbeitsumgebung; **kein** Bericht über den GPU-Rechner.
- `diagnose_design.json`: acht symbolische Designzellen mit offenen Abnahmepunkten; kein ausführbarer Kampagnenplan.

## Offline-Audit wiederholen

Im Vorbereitungsordner, sofern er direkt im vorhandenen DOS-Projektordner liegt:

```bash
python3 audit_existing.py --root ..
```

Das Programm schreibt JSON auf stdout. Exit 0 bedeutet nur: Die ausgewählten historischen Rohdaten bestehen diese begrenzte Integritätsprüfung. Es bestätigt keine neue Experimentbereitschaft. Originaldateien werden nicht verändert.

## Umgebungsprüfung auf dem Messrechner

Die folgende Vorlage erst nach Zuordnung der tatsächlichen Pfade verwenden. `PFAD_ZUR_BISHERIGEN_UMGEBUNG`, `PFAD_ZUM_DOS_PROJEKT`, `PFAD_ZUM_VORBEREITUNGSORDNER` und `PFAD_ZUM_LOKALEN_MODELL` sind Platzhalter:

```bash
PFAD_ZUR_BISHERIGEN_UMGEBUNG/bin/python \
  PFAD_ZUM_VORBEREITUNGSORDNER/inspect_environment.py \
  --root PFAD_ZUM_DOS_PROJEKT \
  --model-dir PFAD_ZUM_LOKALEN_MODELL
```

`--model-dir` ist optional. Ein positiver Dateibefund bestätigt keine Vollständigkeit des Modellcaches. Der Bericht bleibt unabhängig von vorhandenen Paketen bei `experiment_ready: false`, weil Cache-/Budgetwirkung, Kapazitätskontrolle und Messkette gesonderte Abnahmen benötigen.

## Validierung dieser Lieferung

Der Offline-Audit hat 100 Episoden, 2.200 Requests und 131.200 Ausgabetokens geprüft. Keine Integritätsfehler im geprüften Umfang. Sechs gezielte synthetische Checks haben gültige Streams akzeptiert und fehlerhafte Zeitfolgen, Token-IDs, Tokenzahlen, Endzeitpunkte sowie doppelte Request-Identitäten erkannt. Beide Programme wurden auf Syntax geprüft; die Umgebungsprüfung läuft ohne die GPU-Pakete.

Historische Runner, Serverstarter und Studienverträge wurden nicht geändert. Vor dem GPU-Test sind die offenen Schritte im Workflow abzuschließen.

## Übergabe

Den gesamten Ordner auf den Messrechner kopieren. Die Programme sind unabhängig vom ursprünglichen Ablagepfad; den dortigen DOS-Pfad geben wir später als `--root` an. Die historischen Messdaten bleiben im vorhandenen DOS-Repository und werden nicht nochmals in dieses kleine Vorbereitungspaket kopiert. Remote-Zugang wird hier nicht benötigt. Einrichtung, Anpassungen und Messungen gehen wir gemeinsam an, sobald der Nutzer am Messrechner ist.
