# NordSec → ICISSP 2027: fokussierter Offline-Audit (3 ergänzende Auswertungen)

**Stand:** 18.10.2026
**Umfang:** Vorhandene Kontrollkampagne `NordSec_Messergebnisse_20261018.zip`, keine neuen GPU-Messungen, keine Veränderungen der Originaldaten.
**Historischer Forschungsstand:** 192-Episoden-Hauptkampagne, negativer bzw. nicht robuster aggregierter Offload×Burst-Primärbefund bleibt unverändert. Die hier untersuchte 32-Episoden-Kampagne ist eine retrospektive Zusatzvalidierung, kein Ersatz des ursprünglichen Primärtests.

## 1. Provenienz und Audit-Tiefe

- **Input:** 32 vollständige JSONL-Episoden der 32-Episoden-Kampagne (zwei Budgets × acht Wiederholungen × zwei Burst-Bedingungen), acht festgelegte Victims je Episode, 128 identitätsgematchte Burst/No-Burst-Victim-Paare.
- **Verfahren:** Erneutes Parsen der `sse_wire.raw_data`-Payloads einschließlich einzelner Token-IDs und monotone `receive_ns`-Zeitstempel; `progress_trigger.trigger_ns` wird als Nullpunkt verwendet. Die gespeicherten `request_result`-Tokenfolgen dienen ausschließlich der Konsistenzprüfung. Jede einzelne Rohdatei wird SHA-256-gehasht (Dateiliste `source_hashes.csv`).
- **Identität:** Die Paare stimmen in Prompt-Hash, Prompt-Seed und Generierungs-Seed überein. **22 von 128** Paaren erzeugen verschiedene tatsächliche Fortsetzungen; alle bleiben enthalten.
- **Gültigkeit:** 32 von 32 gespeicherten technischen Validitätsflags positiv; alle Victims liefern genau 64 einzeln aufgelöste Token-Receipts. Bei Token 32 liegen beide Bedingungen nach dem Trigger. Unsere Prüfung bestätigt Zeitreihen- und ID-Konsistenz, **nicht** eine unabhängige erneute attestation aller GPU-internen Schedulerereignisse.
- **Statistik:** Acht gepaarte **Blöcke** pro Budget (je Mittel über acht Victims); für jeden Endpunkt 50.000 deterministische Percentile-Block-Bootstrap-Resamples (Seed 20261018; gesondert 20261019 für die Decode-Fluidity-Sensitivität), nicht 64 unabhängige Victim-Replikate. Explorative/deskriptive Intervalle ohne Mehrfachtestkorrektur.
- **Einschränkung:** Die analysierten JSONL-Dateien wurden aus dem vom Nutzer hochgeladenen Messarchiv gewonnen. Eine erneute unabhängige Prüfung der historischen GPU-Ausführung oder des Rechners wird hier **nicht** behauptet.

## 2. Analyse A – vollständige triggerrelative Token-Fortschrittskurve

Definition pro festem Victim-Index \(i\), Episode-Budget \(b\), Wiederholung \(r\) und Tokenposition \(k\):

\[
\Delta_{b,r,i}(k)=[t^{\mathrm{burst}}_{b,r,i,k}-T^{\mathrm{burst}}_{b,r}]-[t^{\mathrm{no\,burst}}_{b,r,i,k}-T^{\mathrm{no\,burst}}_{b,r}].
\]

Innerhalb eines Blocks wird zunächst über die acht Victims gemittelt. Die Bootstrap-Einheit bleibt der Block. Tokenpositionen vor dem Trigger werden in der Grafik nur als diagnostischer Kontext gezeigt.

| Beobachteter zusätzlicher Client-Receipt-Zeitverlust | Budget 512 | Budget 2048 |
|---|---:|---:|
| Trigger → Token 32 | **1.294,40 ms** [1.279,30; 1.309,81] | **1.727,70 ms** [1.714,98; 1.741,07] |
| **Token 32 → Token 64** (Differenz der Effekte) | **461,12 ms** [452,41; 474,61] | **16,43 ms** [7,11; 25,99] |
| Trigger → Token 64 | **1.755,52 ms** [1.743,59; 1.767,52] | **1.744,13 ms** [1.732,68; 1.755,62] |

**Direkter Budgetkontrast 2048 − 512:** Token 32 +433,31 ms [ca. +411 bis +455]; Token 64 −11,39 ms [ca. −31 bis +8]; **Abschnitt Token 32→64 −444,70 ms [−457,74; −431,60].** Der Abschnittsunterschied ist in allen acht gematchten Wiederholungen gleichgerichtet. Die kleine Differenz bis Token 64 ist *kein* Äquivalenzbeweis.

Die vollständige Grafik zeigt für Budget 2048 einen frühen Anstieg des Zusatzverlusts bis etwa Token 23; Budget 512 steigt über mehrere nachfolgende Tokenintervalle an und nähert sich ab ungefähr Token 39 einer ähnlichen Zusatzverzögerung an. Das ist eine kontrollierte **zeitliche Umverteilung der beobachteten Verzögerung**, keine neue Scheduler-Theorie und kein Nachweis eines vollständig gleichen Gesamtzeitverlusts.

**Dateien:** `token_progress_curve.csv` und `token_progress.png` (alle Positionen 16–64, Konfidenzintervalle und Kontrast).

## 3. Analyse B – externe Endpoint-Beeinträchtigung pro angebotener Zusatzlast

Wir berichten bewusst keine „Angreiferkosten“ im wirtschaftlichen oder GPU-bezogenen Sinne. Die angebotene Zusatzlast pro Burst-Episode beträgt **4 × 2048 = 8.192 Input-Tokens**, daneben **4 × 16 = 64 Output-Tokens**. Gleiche Input-Tokens bedeuten nicht generell gleichen GPU-Aufwand über verschiedenes Scheduling hinweg.

Für jedes gepaarte Episodepaar summieren wir die acht *clientseitigen* zusätzlichen Zeitspannen bis zum Empfang von Token 64:

\[
H_{b,r}=\sum_{i=1}^{8}\Delta_{b,r,i}(64)/1000\ \mathrm{seconds}.\qquad
C_{b,r}=1000H_{b,r}/8192.
\]

| Deskriptive Größe | Budget 512 | Budget 2048 |
|---|---:|---:|
| Summierte zusätzliche Endpoint-Zeit je Burst | **14,044 s** [13,949; 14,141] | **13,953 s** [13,861; 14,045] |
| Zusätzliche Victim-Endpoint-Sekunden je 1.000 angebotene Burst-Input-Tokens | **1,714** [1,703; 1,726] | **1,703** [1,692; 1,715] |

Die Summen sind **keine gemessene Server-Beschäftigungszeit**, keine integrierte verlorene Servicekapazität und keine geschätzten monetären Kosten. Parallel laufende Victims können gleichzeitig verzögert werden. Die Größe beschreibt lediglich die summierte Verschiebung der acht vordefinierten Client-Endpoint-Zeitpunkte gegenüber der gepaarten Kontrolle und ist bei festen Victimzahlen/Burst-Tokenmengen vergleichbar. Sie belegt keinen realistisch erreichbaren Angreifer-Trigger.

**Dateien:** `paired_blocks.csv` und `paired_victims.csv`.

## 4. Analyse C – Etalon-Dekodierziele als Sensitivitätsanalyse

### Quellenbasis

[Agrawal et al., *Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems* (2024), §4.1](https://arxiv.org/html/2407.07000v2#S4.SS1) nennt Anwendungs-Decode-Zielwerte **25 ms** (strict), **50 ms** (medium), **100 ms** (relaxed). Die vollständige Etalon-Metrik benötigt daneben eine separat kalibrierte **Prefill-Deadline** in Abhängigkeit von der Promptlänge und verarbeitet Reservezeit (*slack*) mit Deadline-Reset bei Überschreitungen.

### C1. Direkte ITL-Grenzwertüberschreitung

Für jedes Victim betrachten wir alle ITLs, deren **Endtoken nach dem jeweiligen Trigger** beim Client eintrifft. Der erste betrachtete ITL kann den Trigger überbrücken. Die Tabelle gibt die Differenz **Burst minus Kontrolle** des mittleren Victim-Anteils von ITLs über dem Zielwert wieder (in Prozentpunkten; Block-Bootstrap-95%-CIs).

| Ziel-ITL | Budget 512 | Budget 2048 |
|---|---:|---:|
| **25 ms** | **+39,43 pp** [38,54; 40,60] | **+11,22 pp** [9,66; 12,79] |
| **50 ms** | **+36,48 pp** [36,33; 36,61] | **+8,35 pp** [8,34; 8,36] |
| **100 ms** | **+34,33 pp** [34,19; 34,46] | **+8,35 pp** [8,34; 8,36] |

**Interpretation:** Das kleinere Budget führt trotz kleinerer *maximaler* einzelner ITL in diesem Versuchsprofil zu **mehr** ITL-Grenzwertüberschreitungen. Es verteilt Prefill-Interferenz auf mehrere moderate Pausen. Ein kleinerer Maximalstall allein begründet damit keine bessere deadline-basierte Streamingqualität. Das ist eine deskriptive Sensitivität, *keine* vollständige Etalon-Fluidity- oder SLO-Erfüllungsmessung.

### C2. Algorithmus-1-inspirierte Decode-Fluidity (explorativ, nicht Etalon-Index)

Ergänzend wurde die im Original-Etalon (§4.1 Algorithmus 1) angegebene **Slack-Akkumulation und Deadline-Reset-Regel** nur über Decode-ITLs berechnet. Der ersten betrachteten ITL wird `slack=0` zugewiesen; eine Prefill-Deadline wird **nicht** nachträglich geschätzt. Daher ist das Ergebnis **nicht mit veröffentlichten Gesamt-Fluidity-Werten oder tatsächlicher SLO-Erfüllung gleichzusetzen**.

| Decode-Ziel | Budget 512 (Burst; No-burst) | Budget 2048 (Burst; No-burst) |
|---|---:|---:|
| 25 ms | 0,274; 0,998 | 0,385; 1,000 |
| 50 ms | 0,486; 1,000 | 0,573; 1,000 |
| 100 ms | 0,793; 1,000 | 0,757; 1,000 |

Die 100-ms-Reihenfolge ist **nicht robust gegen die Definition des ersten Decode-ITLs**: Wenn ausschließlich Intervalle berücksichtigt werden, deren Anfang und Ende *nach* dem Trigger liegen, betragen die Burst-Werte ungefähr 0,734 (512) und 0,744 (2048). Auch dies spricht gegen eine globale Qualitätsrangfolge anhand dieser partiellen Metrik.

**Dateien:** `deadline_sensitivity.csv`, `deadline_sensitivity.png`, `fluidity_decode_summary.csv`, `fluidity_decode_blocks.csv`, `fluidity_decode_victims.csv`.

## 5. Wissenschaftliche Konsequenz für NordSec

Die drei Analysen sind miteinander vereinbar, ohne eine neue Forschungsrichtung zu erfordern:

1. **Der Budgeteingriff verändert die Verteilung des beobachteten Delay über Output-Tokenpositionen:** Bei 512 tritt ein erheblich größerer Teil nach Token 32 auf; die gematchte Gesamtverzögerung bis Token 64 ist in beiden Konfigurationen ähnlich, ohne getestete Äquivalenzbehauptung.
2. **Der externe Schaden kann mit konstanter angebotener Zusatzarbeit nachvollziehbar normalisiert werden:** rund 14 summierte Victim-Endpoint-Sekunden je Burst, jedoch keine Messung tatsächlicher Angreiferkosten oder serverinterner Arbeit.
3. **Die Metrikauswahl verändert die Bewertung:** Budget 512 verbessert den frühen Fortschritt und Max-ITL, erzeugt aber häufiger Deadline-Schwellenüberschreitungen. Die vollständige Etalon-Fluidity wurde *nicht* nachgewiesen oder widerlegt.

**Vorschlag für das Manuskript:** Die Token-Fortschrittskurve als zusätzliche Hauptabbildung, eine kurze Tabelle mit Early-/Late-/End-Endpoint-Kontrasten, eine knapp formulierte Kostenrelation in der Security-Diskussion sowie die 25-/50-/100-ms-Sensitivität als kontrollierte Zusatzanalyse. Eine Etalon-konforme vollständige Fluidity-Replikation nicht behaupten.

**Grenzen:** nur Llama-3.1-8B, RTX 3090, vLLM 0.17.1, 8 Paare je Budget, 8 aktive Victims, Client-SSE-Receipts, 22/128 unterschiedliche generierte Ausgaben, Trigger mit experimenteller Fortschrittssynchronisierung, keine Kontrolle aller alternativen Scheduler-/Hardwarebedingungen; mehrere explorative Endpunkte ohne Multiplicity-Adjustierung. Insbesondere die präzisen Bootstrap-Intervalle beschreiben die Streuung innerhalb dieser kontrollierten Blöcke, nicht die Generalisierbarkeit.

## 6. Reproduktion

Im beiliegenden Paket `analyze_offline.py` und `analyze_decode_fluidity.py` in denselben Ordner wie die **unveränderte**, separat entpackte `serving_campaign_20261018/` legen und ausführen:

```
python3 analyze_offline.py --root /path/to/serving_campaign_20261018 --out /tmp/nordsec_offline_results
python3 analyze_decode_fluidity.py --root /path/to/serving_campaign_20261018 --out /tmp/nordsec_decode_results
```

Die Output-Pfade müssen neu sein. Abhängigkeiten: NumPy; Matplotlib nur für Diagramme. Es findet **kein** Netzwerkzugriff, kein vLLM-Start und keine Veränderung der historisch gemessenen Dateien statt.
