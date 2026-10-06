# DOS: Reanalyse vorhandener Daten — Chunk-Budget und Warteschlange

Stand: 6. Oktober 2026. Gegenstand sind ausschließlich die zwei von Thomas freigegebenen Bestandsdatenanalysen. Es wurden keine neuen LLM-Requests gesendet, keine GPU-Messungen gestartet und keine Repository-Dateien verändert.

## Ergebnis und Entscheidung

Die Daten liefern einen konkreten Erkenntnisgewinn: Im Budget-Screen steuert das Chunk-Budget die Länge sichtbarer Ausgabepausen deutlich stärker als die zusätzliche Gesamtlaufzeit der zuerst laufenden Requests. Ohne Offload ist die zusätzliche Laufzeit bei Budget 512 und 2048 in diesen drei Wiederholungen praktisch ähnlich; die längsten Ausgabepausen sind bei 512 etwa 70 % kürzer. Mit 12 GiB Offload sinken ebenfalls die Ausgabepausen deutlich, während der Rückgang des zusätzlichen Laufzeiteffekts wesentlich kleiner bleibt.

Die groben Klassen B/C der offiziellen Server-WAITING-Kampagnen passen in sämtlichen 32 Burst-Episoden zum einfachen Wellenmodell des Reviewers. Die zusätzlichen zeitlichen Abstände widersprechen einer FCFS-Erklärung ebenfalls nicht. Das schwächt den Anspruch, die Klassen selbst als eigenständige neue Scheduler-Erkenntnis zu verkaufen. Die Daten beweisen den internen Scheduler-Mechanismus allerdings nicht.

Das ist weder ein Nachweis, dass das gesamte Thema erledigt ist, noch bereits eine ausreichende Novelty-Begründung für ein neues Paper. Der Budget-Befund und die quantitative Wirkung auf spätere Requests sind konkrete Ansatzpunkte. Vor neuen Messkampagnen sollte zuerst geklärt werden, welchen zusätzlichen Beitrag sie gegenüber den nächstliegenden Arbeiten liefern.

## 1. Datenbasis und geprüfte Grenzen

Repository: [TheBuccaneer/DOS](https://github.com/TheBuccaneer/DOS). Festgeschriebener Commit: `cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69`.

| Kampagne | Umfang | Einstellungen |
| --- | --- | --- |
| [Chunk-Budget-Screen](https://github.com/TheBuccaneer/DOS/tree/cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69/new/runs/chunk_budget_screen/results/official_20260716T075455Z) | 36 Episoden, 18 Paare | Llama; Offload 0/12 GiB; Budget 512/1024/2048; Clientparallelität 4; drei Paare je Zelle |
| [Server-WAITING Llama](https://github.com/TheBuccaneer/DOS/tree/cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69/new/runs/server_waiting_confirmation/llama/official_campaign_20260804T141510Z) | 32 Episoden, 16 Paare | Offload 0/12 GiB; Sequenzkapazität 4/8; vier Paare je Zelle |
| [Server-WAITING Qwen, korrigierte Kampagne](https://github.com/TheBuccaneer/DOS/tree/cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69/new/runs/server_waiting_confirmation/qwen/official_campaign_fixed_20260803T183032Z) | 32 Episoden, 16 Paare | Gleiche Matrix wie Llama |

Je Episode: 20 Victim-Requests mit 256 Eingabe- und 64 Ausgabetokens. In der Burst-Bedingung zusätzlich vier Requests mit jeweils 2048 Eingabe- und 16 Ausgabetokens. Frühere Kampagnenkopien, Diagnose-, Stabilisations- und Smoke-Läufe wurden nicht als zusätzliche Wiederholungen aufgenommen. Der Equal-Input-Budget-Control aus dem Manuskript ist nicht Teil dieser Auswertung.

Die vollständigen JSON-Episoden wurden eingelesen. Für die Analyse wurden numerische Zeitfelder, Konfigurationen, Prompt-Hashes und SSE-Zeitstempel mit Tokenanzahlen extrahiert; Text, Token-IDs und große Metrik-Payloads wurden nicht benötigt. Alle 100 Episoden und 50 Vergleichspaare waren vollständig. Konfigurationen und Prompt-/Generierungsidentitäten stimmten innerhalb der Paare überein. 25 Paare liefen mit Kontrolle zuerst, 25 mit Burst zuerst. Für die Budgetkontraste stimmten die Victim-Workloads außerdem innerhalb desselben Repeats über alle drei Budgets überein.

Alle 131.200 ausgegebenen Tokens lagen in den ausgewählten Rohdaten einzeln in SSE-Events vor. Die Tokenzahlen, monotone Zeitreihen und erster/letzter Tokenzeitpunkt wurden geprüft. Der aus den Rohzeitpunkten rekonstruierte TPOT stimmte mit dem gespeicherten TPOT überein. Es gab dabei keine Abweichungen über 0,000001 ms.

Die Hauptkampagne mit 192 Episoden wurde in dieser Runde nicht neu ausgewertet. Sie kann später für die separat vorgeschlagenen Analysen genutzt werden.

**Wichtige Designgrenze:** Der Budget-Screen löst nach dem ersten Token aller vier initialen Requests aus. Beim tatsächlichen Burst-Versand hatten diese Requests bereits ein bis drei Tokens empfangen. Die Haupt- und Server-WAITING-Kampagnen verwenden dagegen den 16-Token-Zeitpunkt. Die Effekte verschiedener Kampagnen wurden deshalb nicht zusammengefasst.

## 2. Chunk-Budget: Was wird tatsächlich verbessert?

### Messgrößen

- **Zusätzliche aktive Laufzeit:** Innerhalb jeder Episode der Median der E2EL der Requests mit Index 0–3, gemessen ab Request-Start. Dann Burst minus Kontrolle innerhalb des Vergleichspaars. Diese Request-Laufzeit enthält nicht die Wartezeit anderer Requests vor ihrem Versand.
- **Größte aktive Ausgabepause:** Für jeden der ersten vier Requests der größte Abstand zwischen zwei aufeinanderfolgenden Tokenempfängen; anschließend der Median dieser vier Maxima. TTFT und der Zeitraum nach dem letzten Token gehören nicht zu dieser Größe. Es handelt sich um eine clientseitig sichtbare Pause.
- **Victim-Gesamtdauer:** Vom ersten Victim-Request-Start bis zum letzten Victim-Stream-Ende. Dieser Episodenzeitraum enthält auch die zeitliche Wirkung des Client-Semaphores auf später gestartete Requests. Er ist keine serverinterne Queuing-Messung.

Pro Zelle werden drei Vergleichspaare zusammengefasst. Die Laufzeitdifferenzen sind **Mediane der drei gepaarten Differenzen**, nicht Differenzen separat aggregierter Mediane. Die Pause-Spalten sind jeweils Mediane der drei Episodenwerte.

| Offload GiB | Budget | Pause Kontrolle ms | Pause mit Zusatzlast ms | Zusätzliche aktive Laufzeit s | Spannweite der drei Laufzeiteffekte s | Zusätzliche Victim-Gesamtdauer s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 512 | 68,1 | 128,6 | 1,657 | 1,633 bis 1,681 | 1,617 |
| 0 | 1024 | 76,2 | 252,7 | 1,726 | 1,721 bis 1,732 | 1,736 |
| 0 | 2048 | 96,8 | 436,1 | 1,672 | 1,664 bis 1,684 | 1,668 |
| 12 | 512 | 2521,5 | 3839,5 | 39,348 | 35,476 bis 39,751 | 39,381 |
| 12 | 1024 | 3294,1 | 6815,3 | 42,139 | 39,839 bis 43,173 | 43,232 |
| 12 | 2048 | 3313,5 | 12543,0 | 43,958 | 41,012 bis 48,003 | 48,686 |

Ohne Offload liegen die zusätzlichen aktiven Laufzeiten bei 512 und 2048 bei rund 1,66–1,67 s. Die Pause mit Zusatzlast steigt dagegen von rund 129 auf 436 ms. Budget 1024 liegt bei der Pause dazwischen, hat aber in allen drei Vergleichskontrasten einen etwas größeren zusätzlichen aktiven Laufzeiteffekt als 2048. Die beobachtete Gesamtlaufzeit verhält sich somit nicht einfach monoton zum Budget.

Mit 12 GiB Offload steigen die Pausen mit Zusatzlast von rund 3,84 über 6,82 auf 12,54 s. Die zusätzlichen aktiven Laufzeiten liegen bei rund 39,35 / 42,14 / 43,96 s. Der Burst ist also auch unter Budget 512 weiterhin mit erheblicher zusätzlicher Laufzeit verbunden. Die No-Burst-Laufzeiten dieser Requests liegen bereits bei rund 97–99 s; ohne Offload bei rund 1,55–1,57 s. Kleinere relative Effekte sind deshalb kein hinreichender Beleg für kleinere absolute Verzögerungen.

### Direkte Kontraste zu Budget 2048

Für dieselbe Offload-Einstellung und denselben Repeat wurden die bereits gepaarten Burst-Effekte verglichen. Beispiel: `(Burst512 − Kontrolle512) − (Burst2048 − Kontrolle2048)`. Negative Werte bedeuten einen kleineren zusätzlichen aktiven Laufzeiteffekt bei dem kleineren Budget. Die Pausen- und Zusatzlastdauer-Prozentwerte sind Mediane der drei repeatbezogenen Verhältnisse; sie werden nicht aus den Medianen der vorigen Tabelle berechnet.

| Offload GiB | Budgetkontrast | Änderung aktiver Laufzeiteffekt s | Spannweite s | Kürzere Pause unter Zusatzlast | Längere Gesamtdauer der vier Zusatzrequests |
| --- | --- | --- | --- | --- | --- |
| 0 | 512 vs. 2048 | -0,015 | -0,050 bis 0,017 | 70,5 % | 10,3 % |
| 0 | 1024 vs. 2048 | 0,049 | 0,042 bis 0,068 | 42,1 % | 6,1 % |
| 12 | 512 vs. 2048 | -4,207 | -12,527 bis -1,664 | 69,4 % | 16,0 % |
| 12 | 1024 vs. 2048 | -1,819 | -8,164 bis 2,161 | 45,7 % | 5,9 % |

Bei 512 versus 2048 ohne Offload reicht der aktive Laufzeitkontrast von etwa −50 bis +17 ms. Daraus lässt sich bei drei Wiederholungen keine gesicherte Gleichwertigkeit ableiten; sichtbar ist eine kleine Änderung gegenüber einem Effekt von rund 1,7 s. Gleichzeitig verkürzen sich die Pausen in jedem Repeat stark. Mit Offload ist der aktive Laufzeitkontrast bei 512 in allen drei Repeats negativ, aber seine Größe schwankt erheblich. Die gesamte Victim-Episodendauer schwankt noch stärker; bei Offload 12 ist sie durch das kleinere Budget nicht in jedem Repeat verbessert.

Der praktische Zielkonflikt ist damit sichtbar: kürzere Unterbrechungen beim Stream können mit fast unveränderter zusätzlicher Victim-Laufzeit und längerer Laufzeit der Zusatzrequests einhergehen. „Schützt besser“ muss deshalb an einem vorher festgelegten Ziel gemessen werden, etwa maximaler Ausgabepause oder Completion-Latenz. Ein Vergleich bloßer Metrik-Rangfolgen belegt noch keine falsche Konfigurationsentscheidung.

**Wissenschaftlicher Wert:** Das beantwortet den Einwand, dass der unmittelbar relevante Konfigurationsparameter nicht untersucht wurde, zumindest als begrenzter Bestandsdaten-Screen. Es begründet eine konkretere Darstellung der zeitlichen Wirkung. Dass kleinere Chunks Ausgabepausen verringern können, wird hier nicht als neuer Mechanismus behauptet. Eine eigenständige Veröffentlichung braucht zusätzlich eine belegte Abgrenzung und Relevanz.

**Statistische Grenze:** Es gibt nur drei Wiederholungen je Zelle. Es wurden keine Signifikanz-, Äquivalenz- oder verallgemeinernden Konfidenzintervall-Claims daraus abgeleitet. Die vielen Einzelrequests/Tokens ersetzen die drei unabhängigen Episodenpaare nicht.

## 3. Warteschlangenmodell: Bleibt ein Widerspruch übrig?

### Zwei Ebenen der Prüfung

Das einfache synchrone Wellenmodell des Reviewers sagt für genau diese Matrix voraus: 20 Requests bei Kapazität 4 enden in einer vollen letzten Welle, also Klasse C; bei Kapazität 8 gibt es eine teilweise letzte Welle, also Klasse B. Diese Vorhersage verwendet die gemessene Burst-Ausgabezeit nicht als Eingabe.

Eine zweite, kontinuierliche FCFS-Plausibilitätsprüfung verzichtet auf exakt gleichzeitige Wellen. Wenn alle 20 Victim-Requests vor der Zusatzlast in derselben Warteschlange stehen, es k Ausführungsplätze gibt und jeder Abschluss einen Platz freigibt, entsteht der erste Platz für einen Zusatzrequest nach Victim-Abschlussrang `20 − k + 1`: bei k=4 Rang 17, bei k=8 Rang 13. Die vier Zusatzrequests könnten nach den Rängen 17–20 beziehungsweise 13–16 nachrücken.

Das ist ein **angenommenes Modell**, kein aus den Daten rekonstruierter vLLM-Admission-Trace. Clientseitige Stream-Enden und letzte Tokenempfänge wurden jeweils als alternative Abschluss-Proxys eingesetzt. Ein Platz kann serverintern bereits vor dem clientseitigen Empfang frei werden; Transport, Pufferung, Prefill und serverinterne Policies bleiben unbeobachtet.

### Ergebnisse aus allen 32 Burst-Episoden

Klasse B: erste Zusatzausgabe nach Ende der initialen dynamischen Kohorte, aber vor Ende aller Victims. Klasse C: erste Zusatzausgabe nach Ende aller Victims. Die folgenden Zeitspalten zeigen Mediane der vier Episoden je Zelle.

| Modell | Offload GiB | k | Beobachtete Klassen | Erste Zusatzausgabe nach modelliertem erstem Platz s | Erste Zusatzausgabe relativ zu letztem Victim-Ende s | Verbleibende Victim-Zeit ab erstem Platz s |
| --- | --- | --- | --- | --- | --- | --- |
| llama | 0 | 4 | {'C': 4} | 0,455 | 0,434 | 0,022 |
| llama | 0 | 8 | {'B': 4} | 0,931 | -2,327 | 3,257 |
| llama | 12 | 4 | {'C': 4} | 13,906 | 12,505 | 1,399 |
| llama | 12 | 8 | {'B': 4} | 29,234 | -115,587 | 144,840 |
| qwen | 0 | 4 | {'C': 4} | 0,426 | 0,405 | 0,021 |
| qwen | 0 | 8 | {'B': 4} | 0,877 | -2,170 | 3,047 |
| qwen | 12 | 4 | {'C': 4} | 14,166 | 12,680 | 1,487 |
| qwen | 12 | 8 | {'B': 4} | 28,787 | -121,349 | 150,136 |

Die Klassen entsprechen in **32 von 32** Episoden der einfachen Vorhersage. Die erste Zusatzausgabe liegt in sämtlichen Episoden nach dem modellierten ersten freien Platz. Die Wahl zwischen letztem Token und Stream-Ende ändert keine einzige Klasse und erzeugt keinen Widerspruch in dieser Proxy-Prüfung.

Bei k=4 liegen die letzten vier Victim-Enden ohne Offload nur etwa 21 ms auseinander; mit Offload etwa 1,4–1,5 s. Die erste sichtbare Zusatzausgabe folgt dem frühesten dieser Enden erst nach etwa 0,43–0,46 s beziehungsweise rund 14 s. In diesen Beobachtungen liegt sie deshalb nach allen Victim-Enden. Exakt synchrone Wellen sind für die beobachtete Klasse C nicht erforderlich.

Bei k=8 bleibt nach Abschlussrang 13 deutlich mehr Victim-Zeit übrig: rund 3,0–3,3 s ohne Offload und rund 145–150 s mit Offload. Die erste Zusatzausgabe erscheint schon nach rund 0,88–0,93 s beziehungsweise rund 29 s. Damit liegt sie innerhalb dieses verbleibenden Zeitfensters: Klasse B.

Das bestätigt die **Plausibilität** der einfachen Erklärung für die grobe Klasse. Es beweist weder, wann der Zusatzrequest intern admitted wurde, noch dass der Zeitraum bis zur ersten Ausgabe ausschließlich Prefill war. Die Rechenmodelle können insbesondere nicht aus der bloßen ersten sichtbaren Zusatzausgabe auf den Beginn von GPU-Arbeit schließen. Zudem wurde N=20 nicht variiert; es gibt keine unabhängige Prüfung an anderen Restklassen N modulo k.

### Was die Klasse nicht beschreibt

Bei k=8 ist der Zeitraum vom 13. bis zum 20. Victim-Abschluss unter Zusatzlast in jedem der vier Repeats länger als in der Kontrolle. Die folgenden Werte sind getrennte Mediane der Episoden-Zeitspannen; ihre Differenz ist kein individueller Request-Effekt.

| Modell | Offload GiB | Kontrolle: Rang 13 bis 20 s | Zusatzlast: Rang 13 bis 20 s |
| --- | --- | --- | --- |
| llama | 0 | 1,545 | 3,257 |
| llama | 12 | 100,043 | 144,840 |
| qwen | 0 | 1,456 | 3,047 |
| qwen | 12 | 105,245 | 150,136 |

Das ist ein deskriptiver Hinweis, dass die quantitative Wirkung im späteren Abschnitt liegt, obwohl die initiale Kohorte in den bisherigen Berichten nahezu unveränderten TPOT aufweist. Es widerspricht FCFS nicht: Das Modell erklärt die Überlappung, unter der weitere Requests langsamer werden können. Die Rang-Zeitspanne allein unterscheidet noch nicht verspäteten ersten Output, langsameren Decode und wechselnde Request-Identitäten. Diese zusätzliche Zerlegung beziehungsweise Kohortensensitivität wurde hier nicht durchgeführt.

**Wissenschaftlicher Wert:** Die Klassentabelle sollte als beobachteter, mit Queue-Geometrie vereinbarer Befund präsentiert werden. Als eigenständige neue Scheduling-Entdeckung trägt sie nach dieser Prüfung wenig. Die spätere quantitative Wirkung ist die konkretere offene Frage.

## 4. Konsequenz für die Weiterarbeit

| Punkt | Bewertung nach der Reanalyse | Konsequenz |
| --- | --- | --- |
| Budgetvergleich fehlt | Vorhandener, vollständiger Screen liegt vor | Ergebnisse einarbeiten; frühes Triggerdesign und n=3 offen ausweisen. Kein neuer Lauf nur zur Wiederholung dieser Bestandsdaten nötig. |
| Kleinere Budgets sind generell schützend | Zu grob: Pause und zusätzliche Laufzeit reagieren unterschiedlich | Zielgröße ausdrücklich wählen; relative und absolute Werte gemeinsam berichten. |
| B/C-Klassen sind eine neue Regime-Erkenntnis | Diese Daten grenzen sie nicht von der einfachen Queue-Erklärung ab | Claim abschwächen und FCFS-/Wellenmodell explizit diskutieren. |
| Es bleibt überhaupt nichts wissenschaftlich Verwertbares | Durch diese Prüfung nicht belegt | Budget-Zielkonflikt und spätere Request-Wirkung gezielt prüfen; Novelty separat abgrenzen. |
| Jetzt große neue Messmatrix starten | Noch nicht begründet | Zuerst die vorhandenen Server-WAITING-Streams nach frühen/späten Abschnitten und stabilen Request-Identitäten auswerten, sofern Thomas zustimmt. |

Mein nächster Vorschlag ist eine begrenzte Bestandsdatenanalyse der später bedienten Requests bei k=8: Welche zusätzliche TTFT-, Decode- und Completion-Verzögerung entsteht dort, und bleibt das Ergebnis bei Vergleichen derselben Prompt-Identitäten erhalten? Danach kann eine konkrete Literaturabgrenzung entscheiden, ob diese Kombination aus Queue-Geometrie, zeitlicher Wirkung und Messziel genügend Eigenständigkeit hat. Erst bei einem positiven Ergebnis wäre eine kleine neue Bestätigungsmessung sinnvoll. Es wird hier keine sichere Nische und keine Annahmegarantie behauptet.

Für die Dissertation lässt sich ein belastbarer Beitrag weiterhin an zeitliche Ressourcengarantien in asynchronen Systemen anbinden. Die thematische Passung allein ersetzt den eigenständigen wissenschaftlichen Beitrag nicht. Eine erneute vollständige Dissertationsbewertung war nicht Teil dieser Runde.

## 5. Reproduzierbarkeit und Grenzen

Das begleitende ZIP enthält den Bericht, `input_compact.json`, den Analysecode, Ergebnis-CSV-Dateien und `results.json`/`validation.json`. Das kompakte Eingabeformat bewahrt die für diese Rechnungen verwendeten Originalzeitstempel; jede Episode trägt ihren Originalpfad und Git-Blob-SHA. Die vollständigen Rohdateien bleiben über den festgeschriebenen Repository-Commit zuordenbar. Der Extrakt ist kein Ersatz für die vollständigen Payloads bei anderen Forschungsfragen.

Analyse erneut ausführen: im mitgelieferten Verzeichnis `python3 analyze_existing.py`. Der Code liest ausschließlich die vorhandene kompakte Eingabedatei und schreibt Ergebnistabellen. Er sendet keine Server-Anfragen. Der Download-Helfer ist bewusst nicht Teil des Auswertungspakets.

Die Prüfung umfasst eine GPU-Klasse, vLLM 0.17.1 und die angegebenen Workloads. Der Budget-Screen verwendet nur Llama und Clientparallelität 4. Die Server-WAITING-Matrix hat vier Paare je Zelle. Die gespeicherten Server-Metrik-Samples liegen in sämtlichen 32 geprüften Burst-Episoden vor dem tatsächlichen Burst-Versand; sie liefern keinen post-Burst-Trace von GPU-Zeit, Admission, Prefill-Start oder KV-Belegung. Die Resultate bleiben Aussagen über die tatsächlich beobachteten clientseitigen Zeitverläufe und die Vereinbarkeit mit einem einfachen Modell.
