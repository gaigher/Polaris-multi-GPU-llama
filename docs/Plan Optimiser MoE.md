# Plan — Optimisations MoE (GPT-OSS 120B / Polaris)

**Projet :** `TheTom/llama-cpp-turboquant`  
**Cible :** Windows 11, 6× AMD Polaris (RX 570/580, 8 Go), Vulkan  
**Modèle :** GPT-OSS 120B Q4_K_M  
**PR cache :** [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401)  
**Runbook lancement / ctx / TQ :** [GPT-OSS_120B.md](GPT-OSS_120B.md)  
**Statut prod :** **S-23** + **`-t 8`** + **`--moe-cache off`** jusqu’à gain net mesuré vs `off`

Ce document est la **source unique** des optimisations MoE : placement, threads, cache Vulkan, HOT_USES, **et** runbook / état PR [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401).

---

## Décision immédiate (prod)

Sur 6× RX 570/580 : `--moe-cache off` + placement **S-23** + `-t 8`.

| Levier | Retenu | Notes |
| ------ | ------ | ----- |
| Placement `-ot` | **S** (fin de bloc), **23** couches GPU | +13 % gen @8k / +9 % @32k vs A |
| Threads | **`-t 8`** | Sweep : 12/16/24 plus lents |
| MoE cache | **`off`** | `512` ~9–10× plus lent (thrash) ; `on`/`auto` refusent staging (T6) |
| HOT_USES | **pas en prod** | Matrice ADMIT∈{1,4,8} gen ~2,6–2,9 < T5-lite ~3,57 < off ~3,9 ; P4 différé |

---

## 1. Objectif

Deux niveaux MoE sur Polaris (VRAM insuffisante pour tout le GGUF) :

1. **Placement statique** (`-ot`) — maximiser le débit des experts résidents.
2. **Cache dynamique** (`--moe-cache`) — n’activer que si gain net vs `off` (aujourd’hui : non).

La PR #401 a corrigé l’**allocation** (faux UMA / aperture ~256 MiB → `device-local + staging`). Le chemin staging reste **beaucoup plus lent** que `off` sur S-23.

Mesure S-23 `@8192` ×3 (virgule = décimale FR) :

| moe-cache | Prompt avg | Gen avg | Ratio vs off |
| --------- | ---------- | ------- | ------------ |
| **off** | ~2,73 tok/s | **~3,35 tok/s** | 1× (réf. ; avec `-t 8` → ~3,9) |
| **512** | ~0,35 tok/s | **~0,37 tok/s** | **~9× plus lent** |

Hits ≈ 49 %, pool ≈ 508 MiB — hits ≠ bénéfice.

---

## 2. Contraintes à préserver

- Windows 11 + Vulkan (pas d’imposer Linux).
- Polaris `wave64` : ne pas forcer `requiredSubgroupSize=32`.
- Ne pas confondre aperture `HOST_VISIBLE` (~256 MiB) avec la VRAM totale.
- Conserver `ggml_backend_vk_lock_queue` / `unlock`.
- Ne pas sacrifier le chemin d’inférence standard pour un cache optionnel.
- `--moe-cache off` = référence perf + repli sûr.
- Changements isolés, mesurables, réversibles.
- Session cache Vulkan = **1 device** (Vulkan0 du scheduler) — pas de multi-GPU moe-cache.

---

## 3. Pourquoi tout ne tient pas en VRAM

~58 Go de poids pour **48 Go** VRAM. Chaque couche d’experts ≈ **1,6 Go** → plafond réaliste **~2–4 couches / carte** (souvent **4**, **3** sur Vulkan0 pour marge compute).


| Config testée | Résultat |
| ------------- | -------- |
| Pas de `-ot` / trop de couches GPU | OOM |
| ~2 couches / GPU + reste CPU | OK (intermédiaire) |
| **3+4+4+4+4+4** | **OK — référence** |
| 5 couches / GPU | OOM poids |
| Tous experts CPU | OK, lent |

Vue utile d’une Polaris :

```text
8 Go
├── poids résidents
├── buffers compute   (~1,1 Go marge observée Vulkan0 @ 8192)
├── KV                (~548 MiB @ 32k, K=q8_0, V=turbo3)
└── espace libre  →  moe-cache (si activé)
```

```text
poids experts GPU  →  calcul GPU          (résident permanent)
poids experts CPU  →  RAM  →  moe-cache  →  VRAM temporaire  →  calcul GPU
```

---

# Partie A — Placement statique et CPU

## A.1 Placement retenu : strat-23 (S)

Mêmes budgets **3+4+4+4+4+4** que A, résidents en **fin de chaque bloc** (Vulkan0 = `0–2` pour marge compute).


| GPU | Couches experts (S) | Buffer modèle |
| --- | ------------------- | ------------- |
| Vulkan0 | **3** (`blk.0–2`) | ~4979 MiB |
| Vulkan1 | **4** (`blk.9–12`) | ~6577 MiB |
| Vulkan2 | **4** (`blk.15–18`) | ~6577 MiB |
| Vulkan3 | **4** (`blk.21–24`) | ~6577 MiB |
| Vulkan4 | **4** (`blk.27–30`) | ~6577 MiB |
| Vulkan5 | **4** (`blk.32–35`) | ~7148 MiB |
| **Total** | **23 / 36** | — |

```powershell
# S (réf.)
-ot "blk\.(3|4|5|6|7|8|13|14|19|20|25|26|31)\.ffn_.*_exps.=CPU"
# A (historique, début de bloc) — plus lent
# -ot "blk\.(3|4|5|6|11|12|17|18|23|24|29|30|35)\.ffn_.*_exps.=CPU"
```

Avec `--split-mode layer`, la carte couche→GPU est fixe : `-ot` ne fait que forcer des experts en CPU. On ne peut pas « envoyer la couche 3 sur Vulkan1 ».

Script incrémental (reload à chaque tour) : `scripts/placer_experts_incremental.ps1`.

## A.2 Moins de couches pour « laisser de la place au cache » — infirmé

Script : `bench_moe_cache_abc.ps1`. CSV : `bench_results/abc-20260927-*.csv`.

#### ctx 8192 (sans TQ)


| Profil | Motif | Couches | Prompt* | Gen* |
| ------ | ----- | ------- | ------- | ---- |
| **A** | 3+4+4+4+4+4 | 23 | **3,08** | **3,66** |
| B | 3+3+4+4+4+4 | 22 | 2,27 | 2,87 |
| C | 3+3+3+4+4+4 | 21 | 2,55 | 3,03 |

**Verdict @ 8192 :** libérer ~1,6–3,2 Go pour le cache **pénalise**. **23 couches** optimal pour le débit court.

#### ctx 32768 (TQ) — 3 runs

| Profil | Couches | Prompt avg | Gen avg |
| ------ | ------- | ---------- | ------- |
| **A** | 23 | 2,18 | **2,55** |
| B | 22 | 2,25 | 2,53 |

**Verdict @ 32768 :** garder **23 couches**. Pas de gain gen reproductible en retirant une couche.

## A.3 Position des couches (même total 23)

| Placement | Ctx | Prompt avg | Gen avg | vs A gen |
| --------- | --- | ---------- | ------- | -------- |
| A début | 8192 | 2,55 | 2,94 | — |
| **S fin** | 8192 | 2,74 | **3,34** | **+13 %** |
| A début | 32768 | 2,28 | 2,64 | — |
| **S fin** | 32768 | 2,44 | **2,87** | **+9 %** |

Variants S / M / H (`abc-20260927-201607`) : **S reste optimal** ; H ≈ S @8k puis −19 % @32k ; M systématiquement plus lent.

**Verdict :** à VRAM égale, **la position compte**. Réf. = **S**.

## A.4 Threads (`-t`)

Sweep `@8192` ×3, S-23, moe-cache=off (`abc-20261004-180910`) :

| `-t` | PromptAvg | GenAvg |
| ---: | --------: | -----: |
| **8** | **3,136** | **3,940** |
| 12 | 3,100 | 3,651 |
| 16 | 2,390 | 2,848 |
| 24 | 2,189 | 3,001 |

Validé aussi `@32768` TQ : gen **~3,53**. **Retenir `-t 8`**. Suite optionnelle : NUMA.

## A.5 Variants V0 à 23 couches

`abc-20261004-190642`, `-t 8`, off :

| Profil | V0 GPU | GenAvg | vs S |
| ------ | ------ | -----: | ---- |
| **S** | 0,1,2 | 3,976 | réf. |
| SV0L | 4,5,6 | 4,001 | +0,6 % (bruit) |
| SV013 / SV023 | frontières | ↓ | abandonnés |

→ **garder S** (V0=`0,1,2`). Confirm SV0L @32k optionnel.

## A.6 Goulot restant

Avec S-23 + off : **13 couches experts sur CPU**. Priorité = placement / CPU (fait) ; cache seulement si marge V0 réelle **et** gen ≥ off.

---

# Partie B — Cache dynamique Vulkan (`--moe-cache`)

## B.1 Historique alloc (résumé)

Avant #401 : faux UMA (aperture ~256 MiB) → host-mapped → `failed to allocate ~508 MiB` même avec VRAM libre. Libérer V5 / V0 (S-22, S-21, S-20) **ne suffisait pas** : session = **Vulkan0 only**.

Après patch : device-local + staging OK (`pools=1`, hits ~49 %) mais **~9× plus lent** (thrash). Gains A→S venaient du **placement `-ot`**, pas du cache (souvent inactif quand free−reserve ≈ 0).

Override diag : `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1`.  
Opt-in staging après T6 : `GGML_VK_MOE_CACHE_ALLOW_STAGING=1` ou budget MiB explicite.

## B.2 Hypothèses à tester (cache)

1. **Thrashing du pool** — pool trop petit → évictions / reloads.
2. **Coût du staging** — copies hôte↔GPU > économie.
3. **Mauvaise politique** — experts gardés ≠ réutilisés bientôt.
4. **Budget mal évalué** — MiB par carte ; marge compute/KV.
5. **Coût de sync** — `WaitIdle` sérialise.
6. `budget_mb == 0` **sticky** — seuil 1 GiB / host-mapped avant budget réel.

Ne pas fixer une cause avant métriques (T3 a ensuite confirmé thrash + WaitIdle).

## B.3 Phase 0 — référence

Par essai : commit + `GGML_VULKAN=ON`, pilote, GGUF, mapping GPU, free VRAM, args (`-ot`, ctx, `-t`, cache), prompt fixe, warmup + ≥5 runs, médiane. PP et gen séparés.

| Variante | Cache | But |
| -------- | ----- | --- |
| A | `off` | référence |
| B | `on` / auto | budget auto |
| C–F | 128 / 256 / 512 / 1024 MiB | capacité vs évictions |

Compteurs : budget demandé/résolu/alloué ; chemin ; hits/misses/évictions ; octets H→D ; temps copie/sync ; fail_alloc ; tok/s.

## B.4 Phase 1 — budget et chemin

- Budget effectif avant création du pool (ou revalidation).
- Si budget > tas `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` → pas de host-mapped.
- Couvrir `budget_mb == 0` (`on`/`auto`).
- `free-minus-reserve` logué ; si free non fiable → conservateur ou auto **off**.
- Sur discret + seul chemin staging : cache **off** par défaut ; log « slow on discrete » (`45bbce5`).

## B.5 Phase 2 — coût du staging

Une optimisation à la fois : réutiliser staging ; regrouper transferts ; éliminer copies redondantes ; async **seulement** si OK matériel. Ne pas retirer les verrous `VkQueue`.

## B.6 Phase 3 — politique de cache

49 % hits ne suffit pas. Comparer : politique actuelle ; LRU ; LFU / récence ; coût reload ; **HOT_USES** (§ B.9). Thrashing → auto peut **désactiver** (réduire le budget peut empirer).

## B.7 Phase 4 — par architecture

| Profil | Stratégie |
| ------ | --------- |
| **Polaris discret** | Prefer `off` ; staging seulement si gain mesuré ; `wave64` OK |
| **UMA / iGPU** | Host-mapped si tas ≥ budget final ; log `host-mapped UMA heap=…` |
| **ReBAR** | Ne pas assimiler à UMA ; mesurer |

## B.8 Ordre de travail cache (T0–T8)

- [x] **T0** — Référence — [`bench_moe_cache_t0_matrix.ps1`](../scripts/bench_moe_cache_t0_matrix.ps1) + [`t0-matrix-20261023-181251`](../bench_results/t0-matrix-20261023-181251.txt). `off` gen_med **3,92** ; `512`/`1024` **~0,38** (~0,10×), staging, hits≈50 %
- [x] **T1** — Budget auto — `vk_moe_prepare_budget` + `vk_moe_revalidate_host_mapped` ([`_publish_github/ggml-vulkan-moe-cache.cpp`](../_publish_github/ggml-vulkan-moe-cache.cpp)). `on` → granted ~1009 MiB, même régression
- [x] **T2** — Compteurs `staging_copies` / `staging_bytes_h2d` / `staging_sync_ns` (`GGML_CUDA_MOE_CACHE_STATS`)
- [x] **T3** — Cause dominante = thrashing + `vkQueueWaitIdle` (~60–120 ms/copy), pas le hit rate
- [~] **T4** — Async / moins de `WaitIdle` — **différé** ; garde thrash (T5-lite) coupe les re-fills
- [~] **T5** — Politiques — **T5-lite fait** ; HOT_USES P2 + matrice ADMIT **pas retenus** (§ B.9)
- [x] **T6** — `on`/`auto` refuse staging discret ; gen ≈ off ; override `ALLOW_STAGING=1`
- [x] **T7** — UMA / ReBAR : **non testable** ici (6× Polaris + ASPEED)
- [x] **T8** — Doc + pack [`_publish_github/`](../_publish_github/COLLER_ICI.txt) pour #401

### Diagnostic T3 (S, 512, 2026-10-24)

| Signal | Valeur | Lecture |
| ------ | ------ | ------- |
| gen tok/s | ~0,38 vs ~3,9 `off` | ~10× plus lent |
| hits | ~49 % | hits ≠ gain |
| fills / évictions | ~21024 / ~20903 | **thrashing** |
| pool | 121 slots / 128 experts | trop petit |
| staging | **~88 GiB** H→D, **~299 s** sync | coût dominant |
| `on` après T6 | refuse staging, pools=0, gen≈off | **safe default** |

**Verdict :** T6 protège la prod. T4/T5 lourds peu rentables tant que working set ≫ pool.

### Validation post-follow-up (#401 `30e6a54` / `fd02aa5`)

Réponse EN : [`REPLY_30e6a54_VALIDATION.md`](../_publish_github/REPLY_30e6a54_VALIDATION.md).

| # | Axe | Statut Polaris | Suite |
|---|-----|----------------|-------|
| V1 | Budget multi-sessions | Documenté (1 session Vulkan0) | [ ] ledger CUDA-like si multi-session |
| V2 | Garde thrash sticky | Mesuré (~3,57 vs ~0,37) | [ ] reset fenêtré si demandé |
| V3 | UMA / budget > 1 GiB | Code OK ; pas de HW UMA ici | [ ] reviewer iGPU |
| V4 | Bench A/B | Fait | [x] |

## B.9 HOT_USES (admission / éviction)

**Statut :** R&D **clos sur Polaris 8 Go** — P2 dans [`_publish_github/ggml-vulkan-moe-cache.cpp`](../_publish_github/ggml-vulkan-moe-cache.cpp) ; matrice ADMIT **pas un win** vs T5-lite / `off`. Prod = `off`.  
Script : [`bench_moe_hot_uses_admit_matrix.ps1`](../scripts/bench_moe_hot_uses_admit_matrix.ps1) (`THROTTLE` = `ADMIT`).

HOT_USES n’est justifié que s’il bat **à la fois** `off` **et** P1 + T5-lite, à budget et charge égaux.

### Déjà en place

| Élément | État |
| ------- | ---- |
| T0–T3, T5-lite, T6 | Fait (ci-dessus) |
| `demand_count` + log top heat | Fait (`GGML_CUDA_MOE_CACHE_STATS`) |
| P2 `admit_after` / `readmit_after` | Fait + matrice |
| P3 `pick_victim(hot_uses)` | common.h ; `heat-evict=0` sur probes |
| P4 vieillissement | **Différé** (peu rentable si pool ≪ working set) |
| Matrice `ADMIT_AFTER` ∈ {1,4,8} | **Faite** 2026-10-25 |

### Instrumentation

| Signal | Où | Notes |
| ------ | --- | ----- |
| Demande miss | `dev.demand_count[key].count++` | effacé après fill OK |
| Hits résidents | `slot.uses++` | top-8 dans `vk_moe_log_stats` |
| Skips admission | `admission_skips` | `demand < admit_after` (ou `readmit_after` si pool plein) |
| Évictions « chaudes » | `heat_evictions` | victim avec `uses > hot_uses` |

### Matrice P2 (S / `512` / 2026-10-25)

| Config | Gen | Hits | Admission | Fills | Staging | Thrash |
| ------ | --- | ---- | --------- | ----- | ------- | ------ |
| `off` | **~3,9** | — | — | — | — | — |
| `512` brut | **~0,37** | ~49 % | — | ~21k | ~88 GiB | n/a |
| `512` + T5-lite | **~3,57** | hits-only | — | ~608 | ~2,5 GiB | 1 |
| admit=2 / th=8 (1er probe) | **~2,89** | ~20 % | ~14k | 605 | ~2,5 GiB | 1 |
| **admit=1 / th=1** | **~2,74** | **4,9 %** | 0 | 608 | ~2,6 GiB | 1 |
| **admit=4 / th=4** | **~2,70** | ~19 % | ~7,9k | 605 | ~2,5 GiB | 1 |
| **admit=8 / th=8** | **~2,61** | ~20 % | ~14,7k | 606 | ~2,5 GiB | 1 |

Logs : [`a1`](../logs/moe-alloc-S-b512-a1-20261025-090319.log), [`a4`](../logs/moe-alloc-S-b512-a4-20261025-090730.log), [`a8`](../logs/moe-alloc-S-b512-a8-20261025-091141.log).

**Verdict :** aucun seuil n’approche T5-lite ni `off`. Warm-up staging toujours ~2,5 GiB puis garde thrash. `admit=1` = pire (churn). **P2 non retenu** ; **P4 non lancé**.

### Politiques

| Id | Description | État |
| -- | ----------- | ---- |
| P0 | Cache off | Réf. perf |
| P1 | Cache + T5-lite | Meilleure config cache mesurée (~3,57 < off) |
| P2 | Admission `admit_after` | Codé ; matrice < P1 |
| P3 | Éviction HOT_USES | Partiel (`pick_victim`) |
| P4 | Vieillissement | Différé |

Pour retester : budget MiB explicite ou `ALLOW_STAGING=1` (T6 refuse `on`/`auto`).

**Retenir** seulement si débit ↑ vs **P1+T5-lite** et **≥ `off`**. Hit rate ↑ sans tok/s ↑ = échec.

### Suite HOT_USES

1. [x] Matrice `ADMIT_AFTER` ∈ {1,4,8} vs P1 — **aucun win**
2. [~] P4 vieillissement — différé (réouvrir si budget ≫ 512 ou UMA)
3. [ ] Scénarios chaud / variable — optionnel si on rouvre
4. [x] Décision Polaris : **`off`** ; T5-lite + T6 = filet si MiB forcé

---

# Partie C — PR #401 (moe-cache Vulkan / faux UMA)

**PR :** [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) — `gaigher:patch-1` → `feature/turboquant-kv-cache`, head [`fd02aa5`](https://github.com/TheTom/llama-cpp-turboquant/pull/401/commits/fd02aa5)  
**Fichier :** `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp`  
**Repo local :** `%USERPROFILE%\Documents\llama-cpp-turboquant`  
**Pack coller :** [`_publish_github/`](../_publish_github/COLLER_ICI.txt)  
PR voisine (subgroup) : § [C.9](#c9-pr-voisine--subgroup-set_rows-wave64) / [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406).

## C.1 Checklist

### Fait

- [x] Diagnostiquer `fail_alloc` ~508 MiB en host-mapped (aperture ~256 MiB)
- [x] Patch : host-mapped seulement si tas UMA ≥ budget ; sinon device-local + staging
- [x] `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1`
- [x] Probe S-23 `512` : pools=1, fail_alloc=0, hits ~49 %
- [x] Bench off vs 512 `@8192` ×3 — **~9× plus lent**
- [x] PR correctness : [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) (éditeur web / fork `gaigher:patch-1`)
- [x] Blocker review : restaurer lock queue `vk_submit_and_wait` — [`45bbce5`](https://github.com/TheTom/llama-cpp-turboquant/pull/401/commits/45bbce5f1a7d68e03b180e17ffd4378a4a669d52)
- [x] Log « slow on discrete » + follow-up T1/T5/T6 — `30e6a54` / `fd02aa5`
- [x] Prod documentée : `--moe-cache off`

### Reste

- [ ] Re-review / merge TheTom

### Hors scope de #401

| Fichier | Action |
| ------- | ------ |
| `ggml-vulkan.cpp` (subgroup SET_ROWS) | PR séparée — § C.9 / [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406) |
| `CMakeLists.txt` (hôte shaders Windows) | Local / PR séparée si besoin |
| `tools/ui/dist.tar.gz*` | **Ne pas committer** |

## C.2 Review TheTom (2026-10-22) + réponses

**OK :** test aperture (plus grand tas `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` vs budget) ; `FORCE_DEVICE_LOCAL`.

**Blocker (corrigé `45bbce5`) :** remettre `ggml_backend_vk_lock_queue` / unlock dans `vk_submit_and_wait` (plusieurs contextes partagent la queue).

| Question | Réponse ferme Polaris |
| -------- | --------------------- |
| Log slow / default off | Mesuré ~9× ; prod = `off` ; log dans `45bbce5` |
| True UMA / ReBAR | **Pas testable** (6× discrets + ASPEED) ; case pour reviewer iGPU |
| `budget_mb == 0` → 1 GiB | Atteignable ; depuis `fd02aa5` prepare + revalidate ; T6 refuse staging pour `on`/`auto` |

## C.3 Contenu PR (référence)

**Titre :** `fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache on discrete AMD`

Correctness only : pas de claim perf. Evidence : avant fail_alloc ; après `UMA heap=256 < need=512` → pool 508 MiB. Note : off ~3352 vs 512 ~369 gen tok/s.

**État git :** head `fd02aa5` (4 commits : `a73fd8a` → `45bbce5` → `30e6a54`/`fd02aa5`). Pack local `_publish_github/ggml-vulkan-moe-cache.cpp` = réf. collée. Tree local Documents peut être vieux — **ne pas** resync aveugle vers GitHub.

Liens : [PR #401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) · [compare base←patch-1](https://github.com/TheTom/llama-cpp-turboquant/compare/feature/turboquant-kv-cache...gaigher:patch-1?expand=1) · procédure coller = pack [`_publish_github/`](../_publish_github/COLLER_ICI.txt) (PRs **manuelles**, pas `gh`).

## C.4 Preuves locales

| Artefact | Chemin |
| -------- | ------ |
| Probe alloc OK | `logs/moe-alloc-S-b512-*.log` |
| CSV probe | `bench_results/moe-alloc-S-*.csv` |
| Bench off vs 512 | `bench_results/abc-20261004-172308*` |
| Scripts | `probe_moe_cache_alloc.ps1`, `bench_moe_cache_abc.ps1` |

```text
[moe-cache] Vulkan: UMA heap=256 MiB < need=512 MiB; using device-local + staging (discrete/aperture path)
[moe-cache] Vulkan session ready (budget=512 MiB, device-local)
[moe-cache] Vulkan0 pool[0]: type=mxfp4 ... total=508 MiB
```

## C.5 Décision tok/s (après bench)

| Résultat | Action |
| -------- | ------ |
| `512` ≥ `off` | Mentionner le gain |
| `512` ≈ `off` | PR correctness ; pas de claim perf |
| `512` ≪ `off` (**cas réel**) | Correctness quand même ; thrash noté ; pistes = budget / HOT_USES |

`@8192` ×3 `abc-20261004-172308` : off gen **3,352** vs 512 **0,369** (~9×).

## C.9 PR voisine — subgroup SET_ROWS (wave64)

**Manuelle** (pas de `gh` / push auto). Issue [#330](https://github.com/TheTom/llama-cpp-turboquant/issues/330) → branche `feature/turboquant-kv-cache`. PR : [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406).

| Élément | Fichier |
| ------- | ------- |
| Procédure coller | [`_publish_github/COLLER_SUBGROUP.txt`](../_publish_github/COLLER_SUBGROUP.txt) |
| Diff | [`polaris-subgroup-only.diff`](../_publish_github/polaris-subgroup-only.diff) / snippet [`SNIPPET_SUBGROUP_SET_ROWS.cpp.txt`](../_publish_github/SNIPPET_SUBGROUP_SET_ROWS.cpp.txt) |
| Corps PR | [`PR_BODY_SUBGROUP.md`](../_publish_github/PR_BODY_SUBGROUP.md) |
| Preuves | [`results/df7f54729/polaris-amd-2.0.279/summary.md`](../results/df7f54729/polaris-amd-2.0.279/summary.md) |
| Contexte | [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md), [Plan wave64 wave32.md](Plan%20wave64%20wave32.md) |

**Titre :** `fix(vulkan): conditional subgroup size for TurboQuant SET_ROWS on wave64-only GPUs`  
**Périmètre :** uniquement `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. Le hunk CMake (`vulkan-shaders-gen`) reste correctif Windows local — hors cette PR.

---

## 4. Non-régression

**Fonctionnel :** `off` intact ; budgets < / = / > aperture ; `0`/auto ; OOM propre ; `FORCE_DEVICE_LOCAL` ; UMA si tas OK ; lock queue ; résultats numériques tolérés ; `wave64`.

**Perf :** médianes PP/gen, ratio vs `off`, hits/évictions/octets ; cache froid après restart.

---

## 5. Critères d’acceptation (cache)

1. Pas d’activation auto sur Polaris si débit < `off`.
2. Budget auto résolu avant alloc dépendante.
3. Chemin + justification dans les logs.
4. Métriques évictions / transferts / sync disponibles.
5. Politique = bénéfice net reproductible.
6. Chemin `off` sans régression significative.
7. Lock queue + `wave64` intacts.
8. UMA / grande aperture validés ou marqués non testés.

**Seuil :** gain > bruit sur ≥5 essais ; fixer ensuite un seuil utile (ex. +5 %).

---

## 6. Ce qu’il ne faut plus prioriser

- Forcer **5 couches** d’experts / Polaris → OOM.
- Retirer des couches pour « activer le cache » (V5, V0, S-21, S-20) → infirmé pour le débit ; alloc corrigée mais thrash.
- Espérer un gain tok/s de `--moe-cache 512` sur S-23 Polaris 8 Go → **infirmé** (~9×).
- Multi-GPU moe-cache Vulkan → **1 device** (Vulkan0).
- Re-bench A/M/H « pour voir » → déjà tranché (S gagne).
- Optimiser le hit rate au détriment du débit.
- Retirer des verrous file pour « gagner » en perf.
- Activer le cache sur la seule VRAM totale annoncée.
- Réécriture complète du backend Vulkan / modifier le KV TurboQuant (sujet ≠ placement MoE).

---

## 7. Prochaines actions (optionnel)

1. Confirm SV0L `@32768` ; P1-NUMA.
2. HOT_USES : **clos** sur Polaris 8 Go (matrice ADMIT sans win ; P4 différé).
3. T4 async WaitIdle — différé.
4. Reviewer iGPU/ReBAR pour V3.

Réf. reste **S + `-t 8` + moe-cache off**.

---

## 8. Références

| Élément | Lien / id |
| ------- | --------- |
| PR cache | [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) |
| Aperture | `a73fd8a` |
| Lock + slow-path log | `45bbce5` |
| Safe defaults / T6 / thrash | `fd02aa5` (aussi `30e6a54`) |
| FORCE device-local | `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1` |
| Allow staging | `GGML_VK_MOE_CACHE_ALLOW_STAGING=1` |
| Probe | `scripts/probe_moe_cache_alloc.ps1` |
| Bench ABC / S | `scripts/bench_moe_cache_abc.ps1` |
| Validation V1–V4 | [`REPLY_30e6a54_VALIDATION.md`](../_publish_github/REPLY_30e6a54_VALIDATION.md) |
| Runbook lancement | [GPT-OSS_120B.md](GPT-OSS_120B.md) |
| Pack coller PR | [`_publish_github/COLLER_ICI.txt`](../_publish_github/COLLER_ICI.txt) |
| Subgroup SET_ROWS | § C.9 / [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406) |
