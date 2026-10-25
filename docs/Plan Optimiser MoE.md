# Optimisations MoE (GPT-OSS 120B / Polaris)

**Cible :** Windows 11, 6× AMD Polaris (RX 570/580, 8 Go), Vulkan  
**Modèle :** GPT-OSS 120B Q4_K_M  
**Runtime :** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) / `llama-server`  
**PR cache Vulkan :** [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401)  
**Runbook lancement :** [GPT-OSS_120B.md](GPT-OSS_120B.md)

**Prod retenue :** placement **S-23** + **`-t 8`** + **`--moe-cache off`**

---

## Décision (prod)

Sur 6× RX 570/580 :

| Levier | Retenu | Notes |
| ------ | ------ | ----- |
| Placement `-ot` | **S** (fin de bloc), **23** couches GPU | +13 % gen @8k / +9 % @32k vs placement A |
| Threads | **`-t 8`** | Sweep : 12 / 16 / 24 plus lents |
| MoE cache | **`off`** | `512` ~9× plus lent (thrash staging) ; `on`/`auto` refusent le staging sur discret |
| HOT_USES | **pas en prod** | Aucun seuil mesuré ne bat `off` ni le garde thrash seul |

---

## 1. Objectif

Deux leviers quand la VRAM ne tient pas tout le GGUF :

1. **Placement statique** (`-ot`) — maximiser le débit des experts résidents.
2. **Cache dynamique** (`--moe-cache`) — n’activer que si gain net vs `off` (sur Polaris 8 Go : non).

La [PR #401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) corrige l’**allocation** (faux UMA / aperture ~256 MiB → `device-local + staging`). Le chemin staging reste **beaucoup plus lent** que `off` sur S-23.

Mesure S-23 `@8192` ×3 :

| moe-cache | Prompt avg | Gen avg | Ratio vs off |
| --------- | ---------- | ------- | ------------ |
| **off** | ~2,73 tok/s | **~3,35 tok/s** | 1× (réf. ; avec `-t 8` → ~3,9) |
| **512** | ~0,35 tok/s | **~0,37 tok/s** | **~9× plus lent** |

Hits ≈ 49 %, pool ≈ 508 MiB — un bon hit rate ne suffit pas.

---

## 2. Contraintes

- Windows 11 + Vulkan.
- Polaris `wave64` : ne pas forcer `requiredSubgroupSize=32` (voir [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md)).
- Ne pas confondre aperture `HOST_VISIBLE` (~256 MiB) et VRAM totale.
- Conserver les verrous `VkQueue` (`lock_queue` / `unlock`).
- `--moe-cache off` = référence perf et repli sûr.
- Session moe-cache Vulkan = **1 device** (Vulkan0) — pas de multi-GPU moe-cache.

---

## 3. Pourquoi tout ne tient pas en VRAM

~58 Go de poids pour **48 Go** VRAM. Chaque couche d’experts ≈ **1,6 Go** → plafond réaliste **~2–4 couches / carte** (souvent **4**, **3** sur Vulkan0 pour marge compute).

| Config testée | Résultat |
| ------------- | -------- |
| Pas de `-ot` / trop de couches GPU | OOM |
| **3+4+4+4+4+4** | **OK — référence** |
| 5 couches / GPU | OOM poids |
| Tous experts CPU | OK, lent |

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

# Partie A — Placement et CPU

## A.1 Placement retenu : strat-23 (S)

Budgets **3+4+4+4+4+4**, résidents en **fin de chaque bloc** (Vulkan0 = `blk.0–2` pour marge compute).

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

Avec `--split-mode layer`, la carte couche→GPU est fixe : `-ot` force seulement des experts en CPU.

Script d’exploration (reload à chaque tour) : `scripts/placer_experts_incremental.ps1`.

## A.2 Moins de couches pour « laisser de la place au cache » — infirmé

Script : `scripts/bench_moe_cache_abc.ps1`.

#### ctx 8192 (sans TQ)

| Profil | Motif | Couches | Prompt* | Gen* |
| ------ | ----- | ------- | ------- | ---- |
| **A** | 3+4+4+4+4+4 | 23 | **3,08** | **3,66** |
| B | 3+3+4+4+4+4 | 22 | 2,27 | 2,87 |
| C | 3+3+3+4+4+4 | 21 | 2,55 | 3,03 |

**Verdict :** libérer ~1,6–3,2 Go pour le cache **pénalise**. Garder **23 couches**.

#### ctx 32768 (TQ)

| Profil | Couches | Prompt avg | Gen avg |
| ------ | ------- | ---------- | ------- |
| **A** | 23 | 2,18 | **2,55** |
| B | 22 | 2,25 | 2,53 |

Pas de gain gen reproductible en retirant une couche.

## A.3 Position des couches (même total 23)

| Placement | Ctx | Prompt avg | Gen avg | vs A gen |
| --------- | --- | ---------- | ------- | -------- |
| A début | 8192 | 2,55 | 2,94 | — |
| **S fin** | 8192 | 2,74 | **3,34** | **+13 %** |
| A début | 32768 | 2,28 | 2,64 | — |
| **S fin** | 32768 | 2,44 | **2,87** | **+9 %** |

**Verdict :** à VRAM égale, **la position compte**. Réf. = **S**.

## A.4 Threads (`-t`)

Sweep `@8192` ×3, S-23, moe-cache=`off` :

| `-t` | PromptAvg | GenAvg |
| ---: | --------: | -----: |
| **8** | **3,136** | **3,940** |
| 12 | 3,100 | 3,651 |
| 16 | 2,390 | 2,848 |
| 24 | 2,189 | 3,001 |

Validé aussi `@32768` TQ : gen **~3,53**. **Retenir `-t 8`**.

## A.5 Goulot restant

Avec S-23 + `off` : **13 couches experts sur CPU**. Priorité = placement / CPU ; n’activer le cache que si marge V0 réelle **et** gen ≥ `off`.

---

# Partie B — Cache dynamique Vulkan (`--moe-cache`)

## B.1 Correctness vs perf

**Avant #401 :** faux UMA (aperture ~256 MiB) → host-mapped → `failed to allocate ~508 MiB` malgré de la VRAM libre. La session moe-cache est sur **Vulkan0 uniquement** : libérer d’autres GPU ne corrigeait pas l’alloc.

**Après #401 :** device-local + staging OK (`pools=1`, hits ~49 %), mais **~9× plus lent** (thrashing + `vkQueueWaitIdle` sur les copies H→D). Les gains A→S viennent du **placement `-ot`**, pas du cache.

Overrides utiles pour diag / force :

| Variable | Rôle |
| -------- | ---- |
| `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1` | Forcer device-local |
| `GGML_VK_MOE_CACHE_ALLOW_STAGING=1` | Autoriser le staging après le refus `on`/`auto` sur discret |

## B.2 Diagnostic (S, budget 512)

| Signal | Valeur | Lecture |
| ------ | ------ | ------- |
| gen tok/s | ~0,38 vs ~3,9 `off` | ~10× plus lent |
| hits | ~49 % | hits ≠ gain |
| fills / évictions | ~21k / ~21k | **thrashing** |
| staging | **~88 GiB** H→D, sync dominante | coût principal |
| `on` / `auto` (post-garde) | refuse staging, gen ≈ `off` | **défaut sûr** |

Scripts de repro : `scripts/probe_moe_cache_alloc.ps1`, `scripts/bench_moe_cache_t0_matrix.ps1`, `scripts/bench_moe_cache_abc.ps1`.

## B.3 HOT_USES (admission / éviction)

Sur Polaris 8 Go, les politiques d’admission (`admit_after`) et d’éviction « chaude » **n’ont pas battu** :

- `off` (~3,9 tok/s gen), ni
- le garde thrash seul (~3,57 tok/s gen, encore sous `off`).

Warm-up staging ~2,5 GiB puis garde thrash. **Prod = `off`.** Script de matrice : `scripts/bench_moe_hot_uses_admit_matrix.ps1`.

## B.4 Profils

| Profil | Stratégie |
| ------ | --------- |
| **Polaris discret** | Prefer `off` ; staging seulement si gain mesuré ; `wave64` OK |
| **UMA / iGPU** | Host-mapped possible si tas ≥ budget ; **non testé** sur cette plateforme |
| **ReBAR** | Ne pas assimiler à UMA ; mesurer à part |

---

# Partie C — PR #401 (moe-cache Vulkan / faux UMA)

**PR :** [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) (`gaigher:patch-1` → `feature/turboquant-kv-cache`)  
**Fichier :** `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp`

## C.1 Périmètre

Correctness : host-mapped seulement si le tas UMA peut tenir le budget ; sinon device-local + staging. Pas de claim de perf sur discret Polaris.

**Titre :** `fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache on discrete AMD`

| Sujet | Statut |
| ----- | ------ |
| Alloc 512 MiB après détection aperture | OK |
| Bench off vs 512 (~9×) | Documenté ; prod = `off` |
| Lock queue (`vk_submit_and_wait`) | Restauré ([`45bbce5`](https://github.com/TheTom/llama-cpp-turboquant/pull/401/commits/45bbce5f1a7d68e03b180e17ffd4378a4a669d52)) |
| Log « slow on discrete » + garde `on`/`auto` | Dans le follow-up PR |
| Merge upstream | En revue |

**Hors scope #401 :** subgroup SET_ROWS → § [C.9](#c9-pr-voisine--subgroup-set_rows-wave64) / [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406).

## C.2 Log attendu (chemin aperture)

```text
[moe-cache] Vulkan: UMA heap=256 MiB < need=512 MiB; using device-local + staging (discrete/aperture path)
[moe-cache] Vulkan session ready (budget=512 MiB, device-local)
[moe-cache] Vulkan0 pool[0]: type=mxfp4 ... total=508 MiB
```

## C.3 Décision tok/s

| Résultat | Action |
| -------- | ------ |
| `512` ≥ `off` | Mentionner le gain |
| `512` ≈ `off` | PR correctness ; pas de claim perf |
| `512` ≪ `off` (**cas réel Polaris**) | Correctness quand même ; documenter thrash ; prod = `off` |

## C.9 PR voisine — subgroup SET_ROWS (wave64)

Sur Polaris, sans correctif, `turbo3` peut assert (subgroup 32 vs wave64).

| Élément | Lien |
| ------- | ---- |
| Issue | [#330](https://github.com/TheTom/llama-cpp-turboquant/issues/330) |
| PR | [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406) |
| Runbook patch / build | [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md) |
| Patch dans ce dépôt | `patches/polaris-subgroup.patch` |

**Titre typique :** `fix(vulkan): conditional subgroup size for TurboQuant SET_ROWS on wave64-only GPUs`  
**Périmètre :** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (pas le hunk CMake Windows local).

---

## 4. Non-régression (cache)

- `off` intact ; budgets < / = / > aperture ; OOM propre.
- `FORCE_DEVICE_LOCAL` / `ALLOW_STAGING` documentés.
- Lock queue + `wave64` intacts.
- UMA / grande aperture : validés ailleurs ou marqués non testés.

**Seuil utile :** gain > bruit sur ≥5 essais avant d’activer le cache en prod.

---

## 5. À ne plus prioriser (sur cette plateforme)

- Forcer **5 couches** d’experts / Polaris → OOM.
- Retirer des couches pour « activer le cache » → infirmé pour le débit.
- Espérer un gain tok/s de `--moe-cache 512` sur S-23 Polaris 8 Go → **infirmé** (~9×).
- Multi-GPU moe-cache Vulkan → **1 device** (Vulkan0).
- Optimiser le hit rate au détriment du débit.
- Retirer des verrous file pour « gagner » en perf.

---

## 6. Références

| Élément | Lien |
| ------- | ---- |
| PR cache | [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) |
| Compare base ← patch | [feature/turboquant-kv-cache…gaigher:patch-1](https://github.com/TheTom/llama-cpp-turboquant/compare/feature/turboquant-kv-cache...gaigher:patch-1?expand=1) |
| Subgroup SET_ROWS | § C.9 / [#406](https://github.com/TheTom/llama-cpp-turboquant/pull/406) |
| FORCE device-local | `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1` |
| Allow staging | `GGML_VK_MOE_CACHE_ALLOW_STAGING=1` |
| Runbook lancement | [GPT-OSS_120B.md](GPT-OSS_120B.md) |
| TurboQuant Polaris | [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md) |
