# Mess- und Auswertevertrag v2

## Beobachtungseinheit und primärer Endpunkt

Decode-Kohorte C, späterer Ausgabetokenindex K und Beobachtungsursprung O vor der Intervention festlegen. Für jeden Request gilt der Empfangszeitpunkt des SSE-Events, mit dem erstmals mindestens K echte Ausgabetoken-IDs vorliegen. Mehrtoken-Events liefern denselben Empfangszeitpunkt für mehrere Tokens; keine Zwischenzeiten interpolieren.

T(C,K) = max Empfangszeitpunkt über C − O, in Millisekunden. Das ist die Zeit bis alle Mitglieder den Index erreichen. Clientempfang ist keine serverinterne Tokenberechnungszeit.

Nicht erreichte Ziele, fehlende Mitglieder, fehlgeschlagene Requests und unvollständige Episoden bleiben ohne numerischen Endpunkt erhalten. Beschädigte Daten erscheinen als invalid_input. Eine Zensierungsanalyse ist nicht implementiert. Diese Version setzt vollständige Blöcke voraus und bricht bei Ausfällen vor inferenzieller Auswertung ab; sie entfernt keine Fälle automatisch.

Vorab festlegen: Client-Queue-/Dispatchdefinition, Triggervergleichbarkeit, Folge bei Nichterreichen, Warm-up-/Cache-Resetprotokoll, unabhängige Replikate, Präzisionsziel und praktische Relevanzschwelle. Der Validator prüft Vollständigkeit, nicht wissenschaftliche Angemessenheit oder den Zeitpunkt der Festlegung.

## Vorhandenes Episodenformat

Wiederverwendet werden `victim_requests`, `raw_sse_events`, `output_token_ids`, `request_start_ns`, `stream_end_ns`, `request_index` und `trigger.trigger_perf_ns`. Request- und Episodenstatus muss für einen beobachteten Primärwert complete sein. Der historische Auditor prüft zusätzlich Prompt-/Seedidentität, erwartete Tokenzahlen und sämtliche SSE-Ereignisse innerhalb der Zeitgrenzen, einschließlich tokenloser Ereignisse.

Die neue Studienerfassung braucht außerdem: vorab dokumentierte Kohorte, Policy und Tenantzuordnung, tatsächlich gesendete Salts, öffentliche Präfixidentität, angebotene Arbeit, tatsächliche Scheduler-/Kapazitätskonfiguration, vollständige Metrikabdeckung und Provenienz. Die Templates sind Anschlussverträge; dieses Paket baut historische Episoden nicht nachträglich zu gültigen Studienepisoden um.

## Cache- und Konfigurationsvertrag

Prefix-Caching in beiden Policy-Stufen aktiv. Shared bedeutet erlaubte Wiederverwendung desselben Präfixes zwischen relevanten Tenants; isolated bedeutet stabiler unterschiedlicher Salt je Tenant. Ein neuer Salt je Request wäre ein anderer Faktor. Salts sind keine Tenant-Speicherquoten. Decoder-Prefixe und Cachezustand des Vergleichs müssen ebenfalls spezifiziert werden.

Dokumentieren: tatsächlich geladene Modellrevision, vLLM-Version, Schedulerklasse, async_scheduling, effektive max_num_batched_tokens und gegebenenfalls max_num_scheduled_tokens, long_prefill_token_threshold, max_num_partial_prefills, max_long_partial_prefills, max_num_seqs, Blockgröße, Blockzahl, KV-Datentyp, Cache-Gruppen und resultierende Kapazität. Nur angeforderte CLI-Werte genügen nicht.

Innerhalb jeder Kapazitätsstufe dieselbe effektive KV-Kapazität über Policies und Budgets. Gleiche gpu_memory_utilization genügt nicht. Der Starter übernimmt kein hypothetisches Kapazitätsniveau aus historischen Logs.

## Metrikvertrag

Metrikantworten mit scrape_start_perf_ns und response_received_perf_ns speichern. Für „vor dem Trigger“ zählt der Antwortempfang; ein vorher gestarteter, später empfangener Scrape ist kein Pre-Trigger-Sample. Zeitvergleiche zwischen Prozessen setzen denselben Host und Boot-/Clock-Kontext voraus. Ferne Maschinen dürfen nicht über perf_counter_ns verglichen werden.

Engine-/Modelllabels erhalten. Fehlende Metriken sind unbekannt. Zählerdifferenzen nur bei vollständiger Serie und ohne Reset ausgeben. Die Zusammenfassung erkennt Abnahmen zwischen Beobachtungen; ein unbeobachteter Reset mit anschließend höherem Zählerstand bleibt ohne Boot-/Engine-Provenienz unerkennbar.

KV-usage ist der Anteil nichtfreier Blöcke, nicht die gesamte GPU-Speicherreservierung. Prefix-Hit-/Queryzähler erlauben Wiederverwendungsdiagnostik. prompt_tokens_by_source erfasst tatsächliche Quellen aggregiert; Context-Iterationslogs liefern zusätzlich den Verlauf. num_preemptions_total ist kumulativ. prompt_tokens_recomputed_total darf nicht als allgemeine Preemption-Recompute-Arbeit interpretiert werden.

Iterationslogging misst keine isolierte GPU-Kernelzeit und kann den Ablauf beeinflussen. Aufwand bei gleicher Konfiguration prüfen. Parser unterstützt den beobachteten vLLM-0.17.1-Logtext für den Ein-Engine-Start. Fehlende Details, Iterationssprünge/Neustarts oder mehrere vermischte Engines verhindern eine saubere Parserabnahme.

## Gepaarte Interaktionsauswertung

H_p(b) = T_zusatz − T_referenz. E_p = H_p(klein) − H_p(groß). Gamma = E_isolated − E_shared. Schutzumkehr verlangt E_shared < 0 und E_isolated > 0.

Die Endpoint-Map enthält pro Beobachtung replicate_id, capacity_label (sufficient/near), cache_policy (shared/isolated), budget_label (small/large), condition (reference/additional), nuisance_config_sha256 und effektive KV-/Budgetwerte. Der Tabellenadapter ergänzt Contract-Hash, Workload-Hash und Modellrevision.

Der Workload-Hash bezeichnet den gemeinsamen vorab eingefrorenen Workloadvertrag; individuelle Prompt-/Seedidentitäten innerhalb der Paare bleiben zusätzlich anhand der Rohdaten zu prüfen. Der Nuisance-Hash enthält unveränderte Einstellungen und schließt die expliziten Policy-/Budget-/Kapazitätsfaktoren aus. Beide Hashes sind Provenienzangaben, keine automatisch bestätigten Laufzeitwerte.

Eine unabhängige Wiederholung umfasst hier alle acht Zellen, jeweils mit einem Referenz-/Zusatzlastpaar. Der Analyzer verlangt mindestens zwei vollständige solche Replikatblöcke, dieselben Replikat-IDs in beiden Kapazitätsstufen und konstante Kapazität je Stufe. Zwei Blöcke genügen zur Berechnung, nicht zur behaupteten belastbaren Präzision. Keine Tokens oder einzelne Requests als unabhängige Replikate zählen.

Der Bootstrap zieht ganze Blöcke und erhält dadurch die Paar-/Faktorkovarianz innerhalb einer Kapazitätsstufe. 95%-Perzentilintervalle sind explorative marginale Intervalle; sie liefern keine simultane Abdeckung und keinen automatischen Umkehr- oder Kausalnachweis. Unsicherheiten einer Differenz zwischen Kapazitätsstufen werden hier nicht berechnet. Das festgelegte Präzisionsziel muss separat bewertet werden.
