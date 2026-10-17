# Übergabe für einen anderen Chat

Du übernimmst die technische Vorbereitung des Forschungsprogramms DOS → ICISSP 2027. Sprache Deutsch, knappe Fortschrittsberichte, Implementierung und tatsächliche Tests statt nur Plantext. Deadline 22. Oktober 2026.

Repository: TheBuccaneer/DOS. Bisher lokaler Pfad `/home/rock/projects/DOS`; zunächst den tatsächlich geöffneten Arbeitsbereich ermitteln. Die neuen Komponenten liegen nach dem Einbau unter `new/scripts/cache_isolation_gate/`, die Dokumentation unter `new/docs/icissp_2027_gate/`. Lies AGENTS.md, EINBAU.md, PACKAGE_STATUS.json, die neuen Verträge und den historischen Bestandsbericht. Dateien unter sources/ und historische Messdaten unverändert erhalten.

Eingefrorene Frage: Verändert tenantweise Prefix-Cache-Isolation die Wirksamkeit kleinerer Scheduling-/Prefill-Budgets zum Schutz unabhängig laufender Decode-Requests, besonders nahe einer kontrollierten KV-Kapazitätsgrenze? Kein neuer DoS-Angriff und keine neue Schedulerarchitektur. Keine Schutzumkehr voraussetzen.

H0: bekannte Cache-Miss-/Prefill-Kosten erklären das Ergebnis. H1: zusätzliche dynamische KV-/Scheduler-Rückkopplung. Alternative: Mess-, Kohorten-, Cache- oder Parameterartefakt. Ein positives Novelty-Urteil ist noch nicht zulässig; später insbesondere den FairCache-Volltextabgleich abschließen. Manuskript und Reviews wurden bisher nicht lokalisiert.

Design: Shared/tenantweise Salts × kleines/großes Budget × ausreichende/nahe KV-Kapazität. Kapazität innerhalb jedes Policy-/Budgetvergleichs konstant; ohne CPU-Weight-Offload. Decode-Kohorte vor der Intervention festlegen. Primärmetrik ist zusätzliche Zeit bis alle vorab bestimmten Mitglieder einen festen späteren Tokenindex erreichen. Clientempfang und serverseitige Ausführung auseinanderhalten.

H ist die gepaarte Zusatzverzögerung. E_p = H_p(klein) − H_p(groß); Gamma = E_isolated − E_shared. Eine Umkehr verlangt E_shared < 0 und E_isolated > 0. Eine positive Gamma genügt nicht.

Jetzt implementiert: korrigierter historischer Auditor, Endpunktextraktor, fortlaufender lokaler Metrikrecorder, Vollsampler-Abschlussadapter, Iterationsparser/-vergleich, Contract-Validator, Endpunkttabellenadapter und gepaarte H/E/Gamma-Auswertung mit Bootstrap ganzer Replikatblöcke. Der Serverstarter ist eine nachvollziehbare Kopie des vorhandenen new/scripts/chunk_budget_screen-Starters. Die begrenzte Cache-Probe verwendet den vorhandenen Prefill-Executor unverändert; Quellenhash wird geprüft.

Die aktuelle Anzahl der bestandenen Tests und die historische Nachprüfung stehen im Paketstatus und Evidenzbericht. Diese Befunde sind keine neuen Studienergebnisse. Alle Testdaten sind synthetische Fixtures. Der echte GPU-/HTTP-Laufzeitnachweis bleibt offen.

Bisher geprüfte Umgebung: vLLM 0.17.1 in `/home/rock/projects/porto/.venv-py313-repair/bin/python3`. NVIDIA-Abfrage war aus dem früheren Ausführungskontext nicht erfolgreich; keinen Treiberdefekt daraus ableiten. Lokaler alter HEAD d1e247ba22b3c505f3b36173655b937f89c2693f, geprüfter GitHub-Stand aee916be70765334218449a569d7629cb69672d7. Aktuellen Stand neu bestimmen.

Wichtige Befunde: alte Starter deaktivierten Prefix-Caching; Payload hatte keine Salts. Der historische WAITING-Sampler speicherte nur den Trigger-Snapshot. Dynamische Kohorten unterschieden sich innerhalb der Paare. Alte Budgetläufe hatten unterschiedliche effektive KV-Kapazitäten. Partielle Prefill-Parameter können indirekt den Threshold verändern. prompt_tokens_recomputed_total ist kein allgemeiner Preemption-Recompute-Zähler.

Nächster Auftrag: Offline-Preflight und Quellenabgleich ausführen, anschließend begrenzte Funktionsprüfungen am lokalen Messsystem. Tatsächlich aufgelöste Schedulerklasse, effektive Budgets, Prefix-Reuse, Kapazität und Metrikabdeckung belegen. Zwei unterschiedliche konfigurierte Zahlen sind kein wirksamer Budgetfaktor. Der Logvergleich liefert Beobachtungen, keine automatische Faktorfreigabe.

Die Vertragsvorlagen enthalten absichtlich offene wissenschaftliche Werte. Diese vor neuen Ergebnissen festlegen, nicht als erledigt erfinden. Die Sampler-Anschlussstellen sind beschrieben, aber im historischen Runner nicht automatisch gepatcht. Ein neuer Kampagnenrunner oder ein autonomer KV-Drucktreiber ist nicht enthalten. Beginne keine Kampagne oder Manuskriptarbeit vor abgeschlossenem technischen Gate.

Berichte strikt getrennt: implementiert; offline getestet; am GPU-Server bestätigt; wissenschaftlich noch offen. Keine endlosen neuen Parameterkombinationen zur Rettung negativer Ergebnisse. Go bei praktisch relevanter reproduzierbarer Interaktion und erfolgreicher unabhängiger Vorhersage; reduzieren bei eigenständigem Interaktionseffekt ohne Umkehr; kill bei nur bekannten Kosten oder unwirksamem Budgetfaktor.
