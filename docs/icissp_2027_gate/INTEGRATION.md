# Anschluss an die vorhandene Messkette

## Quellen und Dateischutz

Die Vorlage ist `new/scripts/chunk_budget_screen/run_server.sh`. Ihre unveränderte Kopie, SHA256 und der vollständige Delta-Patch stehen unter `new/scripts/cache_isolation_gate/reference/`. Die Referenzkopie dient dem Vergleich; wegen ihrer anderen Ordnerposition nicht als ausführbaren Starter verwenden.

`cache_smoke.py` lädt ausschließlich `new/scripts/prefill_confirmation/run_prefill_confirmation.py` und prüft dessen bereits dokumentierten SHA256. Der historische Basisrunner wird weder verändert noch als Kampagnen-CLI aufgerufen. Wiederverwendet werden HttpxTransport, RealClock und execute_completion_request. Der kleine SaltTransport ergänzt den Completion-Payload und begrenzt die Funktionsprobe.

`verify_sources.py` prüft beide Vorlagen. Bei einem abweichenden Hash die neue Quelle fachlich vergleichen; nicht einfach den Hash anpassen, um eine Prüfung grün zu machen. Der Serverstarter ist auf vLLM 0.17.1 begrenzt. Eine neuere Version erfordert erneute Scheduler-/Metrikprüfung.

## Vollständiger bestehender Sampler

Im geprüften WAITING-Runner wird `trigger.metrics_raw_samples = list(metrics_sampler.samples)` am Trigger erzeugt. Nach `await metrics_sampler.stop()` im finally-Block wird dieser Snapshot nicht ergänzt. Deshalb fehlen alle späteren Samples.

Das neue Modul `metrics_contract.py` implementiert `finalize_existing_sampler(episode_result, sampler, clock_domain=...)`. Es stoppt den EXISTIERENDEN Sampler und speichert danach eine tiefe Kopie sämtlicher Samples als `metrics_full_samples`. Der Trigger-Snapshot bleibt erhalten. Auch ein stop()-Fehler hinterlässt die bis dahin vorhandene Reihe und wird erneut ausgelöst.

Wichtig für eine saubere Integration: Das Ergebnisobjekt muss vor dem try/finally-Block angelegt und auch auf frühem Return/Fehlerpfad dasselbe Objekt sein. Einen bereits neu konstruierten Returnwert nachträglich nur über eine lokale Variablenzuweisung zu ändern reicht nicht. Die alte _build_episode_result-Funktion und ihr strenger Schemakontrakt müssen für die neue Studienversion gesondert erweitert werden. Für Abbruchpfade Ergebnisse und Metriken sichern, auch wenn die Episode keinen Primärwert liefert.

Dieses Paket verändert den historischen WAITING-Runner nicht und enthält keinen neuen Interferenzrunner. Der Abschlussadapter ist implementiert und getestet; seine Integration in einen zukünftigen Studienrunner bleibt als eigener überprüfbarer Codeänderungsschritt offen. Alternativ kann record_metrics.py unabhängig beobachten. Dann sind Seitendatei, Episode-ID, Host/Boot-Kontext und vollständige zeitliche Abdeckung ausdrücklich zu verknüpfen.

## Technisches Gate

Die Runtime-Evidenzvorlage listet offene Prüfungen. Zu jeder Prüfung Originaldateien mit Pfad und SHA256 referenzieren: GPU/Modell, Prefix-Reuse, effektive Schedulerwerte, reproduzierbar verschiedene Prefill-Verläufe, gleiche effektive Kapazität, vorab feste Kohorte, vollständige Metriken und Loggingaufwand.

Kein Tool in diesem Paket setzt das Runtime-Gate auf bestanden. Log- oder Contractvollständigkeit allein ist keine Freigabe. Eine belegte Cache-Auslastung allein ist kein KV-Kausalnachweis. Erst nach dem technischen Gate folgt eine separat geplante Acht-Zellen-Diagnose; erst bei reproduzierbarer relevanter Interaktion die Mechanismusprüfung und unabhängige Bestätigung.

Keine weiteren allgemeinen Novelty-Schleifen. Später gezielt CacheSolidarity, PrefixShield, SLOWeave, FairInference, FairCache und When Fancy Eviction Fails mit dem tatsächlich beobachteten Befund vergleichen. H0, H1 und Artefakterklärung ernst nehmen. Negative Befunde nicht durch beliebige neue Parameterkombinationen retten.
