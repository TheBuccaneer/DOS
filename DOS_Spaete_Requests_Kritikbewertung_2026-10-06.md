# DOS: Späte Requests — Beitrag zur Reviewer-Kritik

6. Oktober 2026. Gezielte Fortsetzung der Bestandsdatenprüfung, von Thomas freigegeben. Es wurden keine neuen Server- oder GPU-Messungen gestartet und keine Repository-Dateien verändert.

## Einschätzung

**Diese Auswertung hilft bei der methodischen Überarbeitung. Sie begründet noch keine hinreichende Novelty für ein neues Full Paper.** Die Server-WAITING-Daten enthalten eine konsistente zusätzliche Ausgabezeit bei Kapazität 8, auch wenn man alle 20 Prompt-Identitäten vergleicht. Die ursprünglich betrachtete zuerst aktive Kohorte bildet diese Wirkung nicht vollständig ab. Das stärkt die Aussage, dass eine Betrachtung nur dieser Kohorte keine Bewertung der gesamten Victim-Population erlaubt.

Gleichzeitig ist die individuelle Wartezeit einzelner „später Requests“ schwer interpretierbar: Früh beziehungsweise spät bediente Prompt-Identitäten wechseln stark zwischen den Bedingungen. Ein Vergleich nach Rang und ein Vergleich derselben Prompts beantworten unterschiedliche Fragen. Das ist eine konkrete Grenze vorhandener Claims und kein neuer, durch diese Daten bewiesener Scheduler-Mechanismus.

| Kritikpunkt | Was die Auswertung beiträgt | Urteil |
| --- | --- | --- |
| Ein Teil der gemessenen Verzögerung stammt vom Client-Semaphore | In Server-WAITING gibt es dieses Semaphore nicht. Bei k=8 verlängert sich trotzdem die Ausgabephase, über alle 20 gleichen Prompts betrachtet. | Stärkt die Evidenz jenseits des Client-Semaphore. Die ursprüngliche Closed-Loop-Messung bleibt durch ihr Design begrenzt. |
| Dynamische Kohorten sind nicht identitätsgleich | Die 32 Paare wurden über 640 identische Prompts abgeglichen. Die Ausgabezeitwirkung bleibt populationsweit sichtbar; Aussagen über die individuelle Wartezeit sind deutlich vorsichtiger zu formulieren. | Methodisch besser abgesichert. Die Grenze wird beziffert, nicht wegdiskutiert. |
| Spätere Requests erscheinen nur deshalb unbetroffen, weil sie nicht überlappen | In Server-WAITING k=8 ist die später beginnende Ausgabephase tatsächlich verlängert. | Ergänzt die Closed-Loop-Befunde um vorhandene Daten mit anderer Exposition. Erwartbare Interferenz wird damit nicht automatisch neu. |
| B/C-Klassen können durch einfache Queue-Geometrie erklärt werden | Die quantitative spätere Wirkung ist messbar, widerspricht aber dem einfachen FCFS-Modell nicht. | Der Einwand gegen die Novelty der Klassen bleibt bestehen. |
| Die wissenschaftliche Neuheit ist zu dünn | Die Messungen zeigen jetzt präziser, wer wann betroffen ist und wo Kohortenvergleiche Grenzen haben. | Ein besserer empirischer Befund; eigenständige Neuheit und praktische Relevanz müssen weiterhin konkret abgegrenzt werden. |

## 1. Daten und vorab festgelegte Fragen dieser Runde

Verwendet wurden ausschließlich die beiden offiziellen Server-WAITING-Kampagnen des Commits `cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69`: Llama `official_campaign_20260804T141510Z` und Qwen `official_campaign_fixed_20260803T183032Z`. Umfang: 64 Episoden, 32 Burst/Kontroll-Paare, vier Paare pro Modell × Offload × Kapazität. Die vollständigen Rohdaten waren bereits in der vorigen Runde eingelesen worden.

Die Fragen lauten: Ist die im späteren Abschnitt sichtbare Verlängerung eine Verlängerung der Zeit bis zum ersten Token oder der anschließenden Ausgabe? Bleibt sie bestehen, wenn die Prompt-Identitäten übereinstimmen? Ändert das die Bewertung der bisherigen Kritik? Diese Analyse ist **explorativ nach Ablehnung und Voranalyse**, keine nachträgliche Bestätigung einer ursprünglichen Präregistrierung.

Alle 640 Request-Vergleiche stimmen in Prompt-Hash, Prompt-Seed und Generierungs-Seed überein. Die Prompt-Hashes sind innerhalb jeder Episode eindeutig. Die Ausgabezahl ist jeweils 64 Tokens. Die 640 Requests sind korreliert; statistische Wiederholungseinheit sind die vier Episodenpaare je Zelle.

## 2. Phasenzerlegung und Kohorten

Für jeden Request wurde gerechnet:

`Completion = TTFT + Ausgabespanne + Abschlussrest`

TTFT ist erster Tokenempfang minus Request-Start. Die Ausgabespanne ist letzter minus erster Tokenempfang. Der Abschlussrest ist Stream-Ende minus letzter Tokenempfang. Zusätzlich wurde die Zeit von der Task-Erzeugung bis zum Request-Start berücksichtigt. Die Zerlegung schließt für jeden Request und für die Gruppenmittelwerte numerisch bis auf weniger als 0,000000001 s.

**TTFT ist keine isolierte serverinterne Wartezeit.** Sie umfasst unter anderem Transport, Queueing und Prefill. Die Ausgabespanne ist clientseitiger Empfangsverlauf, keine gemessene GPU-Decode-Zeit. Aus diesen Daten lässt sich die zusätzliche interne Queue-Zeit nicht separat identifizieren.

Verglichen wurden:

- alle 20 gleichen Prompt-Identitäten als primärer, von einer Auswahl früher/später Identitäten unabhängiger Vergleich;
- die in der Kontrolle zuletzt beginnenden vier Prompts, mit denselben Prompts unter Zusatzlast;
- die unter Zusatzlast zuletzt beginnenden vier Prompts, mit denselben Prompts in der Kontrolle — eine ausdrücklich deskriptive, nach der Zusatzlast definierte Gruppe;
- die jeweils letzten vier Ausgabestarts beider Bedingungen, bei denen die Prompts unterschiedlich sein können;
- als zusätzliche Sensitivität die letzten vier Request-Starts der Kontrolle mit ihren identischen Prompts unter Zusatzlast.

„Spät“ bedeutet hier Ausgabestart-Rang 17–20, nicht Request-Index 16–19 und nicht ein behaupteter interner Admission-Rang. Die gemeinsame Population aller 20 Prompts vermeidet die wesentliche Unsicherheit einer selektierten kleinen Kohorte. Die feste Auswahl nach Kontrolle vermeidet eine Auswahl ausschließlich aus der Burst-Bedingung, ist aber selbst nach Kontroll-Ergebnis selektiert; Regression zum Mittelwert und wechselnde Bedienreihenfolge sind weiterhin relevant.

Alle folgenden Effekte sind innerhalb eines Episodenpaars gebildete Unterschiede von Request-Mittelwerten und anschließend das arithmetische Mittel der vier Paare. Dadurch bleiben die Phasen additiv. Es handelt sich nicht um die Medianwerte der ursprünglichen Papierfiguren.

## 3. Wirkung auf alle 20 identischen Prompts

| Modell | Offload GiB | k | Zusätzliche TTFT s | Zusätzliche Ausgabespanne s | Zusätzliche Completion s | Spannweite Completion-Effekt der vier Paare s | Positive Paare |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 0 | 4 | -0,002 | -0,004 | -0,006 | -0,021 bis 0,003 | 2/4 |
| llama | 0 | 8 | 0,029 | 0,317 | 0,346 | 0,337 bis 0,357 | 4/4 |
| llama | 12 | 4 | 2,024 | 0,013 | 2,037 | -0,024 bis 8,181 | 2/4 |
| llama | 12 | 8 | 1,738 | 8,222 | 9,960 | 4,163 bis 18,042 | 4/4 |
| qwen | 0 | 4 | 0,001 | 0,001 | 0,002 | -0,007 bis 0,012 | 2/4 |
| qwen | 0 | 8 | 0,032 | 0,284 | 0,316 | 0,309 bis 0,321 | 4/4 |
| qwen | 12 | 4 | 0,002 | 0,001 | 0,003 | -0,007 bis 0,012 | 3/4 |
| qwen | 12 | 8 | 1,905 | 8,061 | 9,965 | 8,969 bis 12,930 | 4/4 |

Bei k=8 ist der Completion-Effekt in allen 16 Episodenpaaren positiv. Ohne Offload entsteht im Mittel je Request ein zusätzlicher Completion-Effekt von etwa 0,32–0,35 s; mit Offload von etwa 9,96–9,97 s. Der größte Anteil liegt jeweils in der Ausgabespanne: etwa 0,28–0,32 s beziehungsweise 8,06–8,22 s. Die Zeit bis zum ersten Token trägt im populationsweiten Mittel deutlich weniger bei; der Abschlussrest ist hier praktisch vernachlässigbar.

Damit ist die zusätzliche Ausgabezeit unabhängig davon sichtbar, welcher konkrete Prompt zur jeweils späteren Gruppe gehört. Das belegt in dieser Kampagne eine veränderte Nutzerwirkung über die ganze Victim-Population, die die zuerst aktive Kohorte allein nicht repräsentiert.

Bei k=4 ergibt sich kein gleichermaßen konsistentes Muster. Llama mit Offload 12 enthält einen einzelnen deutlich positiven Completion-Effekt von rund 8,18 s, überwiegend vor dem ersten Token; die anderen drei Paare liegen ungefähr zwischen −0,024 und +0,002 s. Der Zellenmittelwert von etwa +2,04 s ist deshalb keine stabile Vierfach-Replikation. Dieses Paar wurde nicht ausgeschlossen. Aus dem frühen Abschnitt allein wäre ein solcher späterer Unterschied ebenfalls nicht zu erkennen; seine Ursache ist mit den vorhandenen internen Traces nicht zu bestimmen.

## 4. Ausgabezeit der unter Zusatzlast spät bedienten vier Prompts

Hier werden die Prompts der Burst-Ränge 17–20 mit ihren jeweils identischen Kontroll-Prompts verglichen. Die Gruppe ist nach der Burst-Bedingung definiert und dient der Beschreibung tatsächlich später Streams, nicht als unverzerrt vorab definierte kausale Untergruppe.

| Modell | Offload GiB | Zusätzliche Ausgabespanne je Request s | Spannweite der vier Paare s | Positive Paare |
| --- | --- | --- | --- | --- |
| llama | 0 | 1,473 | 1,363 bis 1,556 | 4/4 |
| llama | 12 | 38,765 | 35,105 bis 42,509 | 4/4 |
| qwen | 0 | 1,345 | 1,248 bis 1,445 | 4/4 |
| qwen | 12 | 38,900 | 35,790 bis 42,968 | 4/4 |

Die zusätzliche Ausgabespanne bleibt in jedem der vier Paare pro Zelle positiv. Ohne Offload beträgt sie etwa 1,34–1,47 s; mit Offload etwa 38,8–38,9 s. In sämtlichen 64 unter Zusatzlast spät beginnenden Requests bei k=8 überschreitet die Ausgabespanne sogar das Maximum aller 20 Kontroll-Requests des zugehörigen Paars. Der Befund kann somit nicht allein daraus entstehen, dass zufällig andere Prompts verglichen werden.

Diese Relation beweist jedoch nicht, dass der gruppenspezifische Effekt für vorab ausgewählte Requests ebenso groß wäre. Die Zusammensetzung früher/später Gruppen ist selbst ein Teil des beobachteten Laufablaufs. Dass Streams, die gleichzeitig mit weiterer Arbeit laufen, langsamer werden, bleibt außerdem mit dem bekannten Interferenzmechanismus vereinbar.

## 5. Wo die Wartezeitaussage instabil wird

Die letzten vier Prompts beider Bedingungen stimmen selten überein:

| Modell | Offload GiB | Identische späte Prompts je Repeat, von 4 | Paare mit vollständig gleicher später Gruppe | Späte Burst-Streams über Kontrollmaximum |
| --- | --- | --- | --- | --- |
| llama | 0 | 0, 0, 0, 1 | 0/4 | 16/16 |
| llama | 12 | 3, 0, 0, 0 | 0/4 | 16/16 |
| qwen | 0 | 0, 0, 0, 0 | 0/4 | 16/16 |
| qwen | 12 | 3, 1, 0, 0 | 0/4 | 16/16 |

Bei k=8 haben 12 der 16 Paare überhaupt keinen gemeinsamen Prompt in der jeweils späten Vierergruppe. In keinem der 16 Paare ist die späte Vierergruppe vollständig identisch. Die sichtbare Position in der Bedienreihenfolge variiert auch in den k=4-Paaren.

Wählt man stattdessen die **in der Kontrolle späten vier** fest aus und verfolgt diese gleichen Prompts unter Zusatzlast, ergibt sich:

| Modell | Offload GiB | TTFT-Effekt s | Ausgabespannen-Effekt s | Completion-Effekt s |
| --- | --- | --- | --- | --- |
| llama | 0 | -1,975 | 0,192 | -1,782 |
| llama | 12 | -107,691 | 10,013 | -97,677 |
| qwen | 0 | -1,853 | 0,075 | -1,777 |
| qwen | 12 | -106,307 | 11,915 | -94,392 |

Die großen negativen TTFT- und Completion-Werte sind kein Nachweis einer Schutzwirkung der Zusatzlast. Die in der Kontrolle spät gestarteten Prompts laufen in der anderen Episode vielfach früher. Die Auswahl anhand hoher Kontroll-Ränge und die wechselnde Bedienreihenfolge machen diesen Untergruppenvergleich anfällig. Auch die Sensitivität nach festen Dispatch-Identitäten zeigt uneinheitliche individuelle Completion-Effekte.

Die Daten zeigen keine isolierte kausale Erklärung, warum die Prompt-Reihenfolge wechselt. Insbesondere ist nicht belegt, dass die Zusatzlast selbst dieses Umordnen verursacht; unabhängige Episoden können bereits unterschiedliche Ankunfts- und Bedienreihenfolgen besitzen. Eine Behauptung über ein einzelnes Prompt-Schicksal ist deshalb wesentlich schwächer abgesichert als die zusätzliche Ausgabespanne der Gesamtpopulation.

## 6. Konkrete Konsequenz für das Paper

1. **Server-WAITING um einen populationsweiten, identitätsgeprüften Effekt ergänzen.** Die Kapazitätsklassen allein reichen nicht: Für k=8 sollte die zusätzliche TTFT-/Ausgabespannen-/Completion-Wirkung aller 20 Victims mit den vier Einzelpaaren sichtbar werden.
2. **Die spätere Ausgabephase ausdrücklich zeigen.** Die unter Zusatzlast spät gestarteten Streams haben eine größere Ausgabespanne auch gegenüber ihren eigenen Kontroll-Prompts. Die Gruppe und ihre nachträgliche Auswahl müssen offengelegt werden.
3. **Individuelle Wartezeitclaims begrenzen.** „Die letzten vier warten zusätzlich X Sekunden“ ist ohne Identitäts- und Reihenfolgebezug nicht belastbar. Zulässig sind sauber definierte Populations- und Ranggruppen-Aussagen mit ihren jeweiligen Grenzen.
4. **Den FCFS-Einwand weiter anerkennen.** Die zusätzliche Streaming-Wirkung passt zur erwartbaren Überlappung mit der Zusatzlast und widerlegt das einfache Queue-Modell nicht.
5. **Novelty nicht aus einer weiteren Metrik ableiten.** Diese Auswertung erhöht die Aussagekraft der vorhandenen Experimente. Die verbleibende Veröffentlichungshürde ist eine präzise zusätzliche Erkenntnis gegenüber dem Stand der Forschung, einschließlich ihrer praktischen Relevanz und Übertragbarkeit.

**Reviewer-Urteil für diese Runde:** Ein sinnvoller, auf konkrete Kritik gerichteter Ausbau der Evidenz. Eine belastbare Ergänzung der Überarbeitung; kein Grund, die ursprüngliche Ablehnung jetzt allein deshalb in „annehmen“ zu ändern. Vor einer neuen Messkampagne wäre eine enge Literaturabgrenzung des nun konkreten Beitrags sinnvoller als weitere beliebige Parameterreihen.

## Reproduzierbarkeit

Das ZIP enthält diesen Bericht, die kompakten Daten der 64 Server-WAITING-Episoden, `analyze_late_requests.py`, die identitätsbezogenen Einzelwerte, Gruppeneffekte und Kohortenprüfungen. Ausführen: `python3 analyze_late_requests.py` im entpackten Verzeichnis. Die vier Episodenpaare je Zelle ersetzen keine breite Generalisierung; es wurden keine Signifikanz- oder Annahmewahrscheinlichkeiten berechnet.

Quellen: [Llama-Kampagne](https://github.com/TheBuccaneer/DOS/tree/cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69/new/runs/server_waiting_confirmation/llama/official_campaign_20260804T141510Z), [Qwen-Kampagne](https://github.com/TheBuccaneer/DOS/tree/cb1548fef27a86d7b5b47ba069c0b2ce9fff6e69/new/runs/server_waiting_confirmation/qwen/official_campaign_fixed_20260803T183032Z). Der Manuskriptvergleich betrifft insbesondere Abschnitt 4.4, der bereits auf unterschiedliche dynamische Request-Identitäten hinweist. Diese Analyse beziffert diese Grenze; sie entdeckt keinen zuvor verschwiegenen Fehler.
