# Plan — Optimiser PP / TG / VRAM sur 6× Polaris

**Statut (2026-11-12) :** priorités 1–2 **closes** — MMQ + `-b 1024 -ub 256`. Hors TurboQuant / MoE (déjà avancés).  
**Principe :** optimiser **simultanément** débit de génération (TG), préremplissage (PP) et occupation VRAM. Un gain sur un axe peut dégrader fortement les deux autres.

**Référence plateforme :** Llama 3.3 70B Q4, `--split-mode layer`, ctx 25 600, 5× RX 580 + 1× RX 570, Windows Vulkan.  
**Campagnes séparées :** A = Llama 70B dense · B = GPT-OSS 120B MoE (S-23, `-t 8`, `--moe-cache off`).

Source externe : [doteyeso-ops/rx580-vulkan-fa-lds](https://github.com/doteyeso-ops/rx580-vulkan-fa-lds) — +13,36 % PP sur 1× RX 580.  
**Résultat local 6× Polaris (2026-11-02) :** matrice 4 builds → **MMQ +14,0 % PP**, FA ≈ 0, A+B ≤ MMQ. Détail : [`../bench_results/polaris-llama70-20261102-195848.summary.md`](../bench_results/polaris-llama70-20261102-195848.summary.md). Binaire de travail : `builds/mmq`.

---

## Objectifs triaxiaux

| Axe | Indicateur | Attention |
| --- | ---------- | --------- |
| **PP** | tok/s prompt (`prompt_per_second`) | Batchs, FA, MMQ, threads `-tb` |
| **TG** | tok/s génération (`predicted_per_second`) | Bande passante mémoire, quant, split, spéculation |
| **VRAM** | buffers par GPU + OOM | `-np 1`, TQ, quant KV, `-b`/`-ub`, tensor-split |

Méthode : **même session**, warm-up, ≥ 5 répétitions, variantes **alternées**, contrôle null si possible (même binaire, deux bras). Dérive inter-session mesurée ailleurs jusqu’à **+21,7 %** — ne pas comparer des builds testés à des jours différents.

Script : [`scripts/benchmark_polaris.ps1`](../scripts/benchmark_polaris.ps1).

---

## Priorité 1 — Patch Vulkan GCN (FA + MMQ) — FAIT

**Clos (2026-11-02).** Retenir MMQ seul. FA non utile sur Llama 70B (têtes trop petites pour la branche GCN). Les conditions existent dans TheTom `feature/turboquant-kv-cache` (`ggml-vulkan.cpp`).

### A. Flash Attention — LDS / occupancy

```diff
- if (... maxComputeSharedMemorySize == 65536) {
+ if (... maxComputeSharedMemorySize >= 32768) {
```

Sur Polaris, `maxComputeSharedMemorySize` = **32 768** (pilote AMD propriétaire). La condition `== 65536` **exclut** donc la branche GCN d’occupation FA. Avec `>= 32768`, la limitation d’occupation GCN (`n_rows <= 8`, `hsk >= 256`) peut s’appliquer.

**Caveat 70B :** Llama 3.3 70B a des têtes plus petites ; la branche GCN FA n’est pas forcément sollicitée. Mesurer quand même.

### B. MMQ — warptiles GCN sous pilote propriétaire

```diff
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

Sous Windows + pilote propriétaire, le réglage warptile MMQ GCN est **aujourd’hui sauté**. C’est la piste la plus intéressante pour le 70B Q4 dense.

### Protocole 4 builds (ne pas confondre les hunks)

| Build | FA | MMQ | Patch |
| ----- | -- | --- | ----- |
| Référence | original | original | aucun |
| A | corrigé | original | `patches/gcn-fa-lds.patch` |
| B | original | corrigé | `patches/gcn-mmq-proprietary.patch` |
| A+B | corrigé | corrigé | `patches/gcn-fa-mmq-combined.patch` |

```powershell
.\scripts\appliquer_patch_gcn.ps1 -Variante fa      # build A
.\scripts\appliquer_patch_gcn.ps1 -Variante mmq     # build B
.\scripts\appliquer_patch_gcn.ps1 -Variante both    # build A+B
```

Mesurer Llama 70B **et** GPT-OSS séparément. L’essai externe porte sur Qwen MoE 1× RX 580 — confirmé en tendance sur 6× Polaris pour **MMQ** (+14 % PP), pas pour FA.

| Build | PP | TG | Verdict 70B 6 GPU |
| ----- | -- | -- | ----------------- |
| ref | 13,60 | 1,18 | baseline |
| fa | 13,55 (−0,4 %) | 1,19 | inutile |
| **mmq** | **15,51 (+14 %)** | 1,20 | **retenu** |
| both | 15,33 (+12,7 %) | 1,16 | ≤ mmq |

---

## Priorité 2 — Batchs PP (`-b` / `-ub` / `-t` / `-tb`) — FAIT (-b/-ub)

Sur le binaire **`builds/mmq`**. Banc 2026-11-12 (`polaris-llama70-20261112-115638`) : **retenir `-b 1024 -ub 256`** (PP 16,88, +1,7 % vs 2048:512). `4096:1024` −4,5 %. TG plat. `-tb` encore à balayer.

| Test | `-b` | `-ub` |
| ---- | ---- | ----- |
| Réf. | 2048 | 512 |
| A | 1024 | 256 |
| B | 2048 | 256 |
| C | 4096 | 512 |
| D | 4096 | 1024 |

Puis threads (réf. génération `-t 8` déjà établie pour GPT-OSS) :

| `-t` | `-tb` |
| ---- | ----- |
| 8 | 8 |
| 8 | 12 |
| 8 | 16 |
| 8 | 24 |
| 12 | 16 |

Objectif principal : **PP**, pas TG. Valeurs élevées → risque VRAM temporaire / OOM — monter progressivement.

Exposé aussi dans `lancer_70b_6gpu.ps1` (`-Batch`, `-UBatch`, `-Threads`, `-ThreadsBatch`).

---

## Priorité 3 — `--tensor-split` (répartition de couches)

Toujours en `--split-mode layer`. Ce n’est **pas** du parallélisme tensoriel : ce sont des **proportions** de couches / KV.

| Config | Exemple (RX 570 en Vulkan0) |
| ------ | --------------------------- |
| Auto | répartition actuelle |
| A | `0.85,1,1,1,1,1` |
| B | `0.75,1,1,1,1,1` |
| C | `1,1,1,1,1,0.85` (si RX 570 en dernier) |

Hypothèse : la carte la plus lente / la plus juste en VRAM bride la chaîne layer.

---

## Priorité 4 — Ordre PCIe / NUMA des GPU

Carte mère ASUS Z10PA-D8 (2 sockets). Cartographier avant de changer `--device` :

- largeur / génération PCIe **négociée sous charge**
- rattachement socket / nœud NUMA
- fréquences, températures, VRAM libre (GPU-Z / HWiNFO)

Regrouper les GPU du même domaine PCIe / socket peut réduire le coût des transferts inter-couches.

---

## Priorité 5 — NUMA / affinité CPU (campagne B — GPT-OSS)

13 couches d’experts sur CPU → bande passante DRAM et placement NUMA comptent.

| Test | Config |
| ---- | ------ |
| Réf. | actuelle (S-23, `-t 8`, moe-cache off) |
| A | `--numa distribute` |
| B | `--numa isolate` |
| C | isolate + affinité sur **un** socket |

Ne pas relancer le balayage `-t` 12/16/24 déjà moins bon en S-23 : on change la **localisation**, pas le nombre de threads.

---

## Priorité 6 — Quantification GGUF (70B)

| Quant | Hypothèse |
| ----- | --------- |
| Q4_K_M | référence |
| Q4_K_S | un peu plus léger |
| IQ4_XS | autre 4 bits (shaders Vulkan ≠ forcément aussi rapides) |
| Q3_K_M | moins de bande passante |
| Q5_K_M | coût précision |

Comparer **PP + TG + qualité** (perplexité / récupération long ctx). Prioriser Q4_K_S et IQ4_XS avant Q3.

---

## Priorité 7 — Décodage spéculatif (TG, après Vulkan)

- Draft CPU : Llama 3.2 1B + `--spec-type draft-simple`, `--spec-draft-n-max` 3 puis 5
- Sans 2ᵉ modèle : `--spec-type ngram-simple` puis `ngram-mod`

Mesurer taux d’acceptation **et** TG net. N’améliore en général pas le PP.

---

## Priorité 8 — Fréquences mémoire / core GPU

Un facteur à la fois (sinon confondu) :

| Test | Modif |
| ---- | ----- |
| A | stock |
| B | mem +50 MHz |
| C | mem +100 MHz |
| D | core +50 MHz |
| E | core +100 MHz |

Suivre stabilité, températures, **tok/s par watt** (2 PSU).

---

## Plus tard — `--split-mode tensor`

Essai ponctuel avec `-fa on`. Contraintes : beaucoup plus de trafic PCIe ; KV quantifié souvent incompatible (F16 requis). Attendu : `layer` reste meilleur sur Polaris PCIe. Non prioritaire.

---

## Campagnes de mesure

### Campagne A — Llama 3.3 70B

- PP : ctx prompt 4k / 8k / 16k / 25,6k
- TG : contexte déjà rempli, **128** tokens générés
- Axes : Vulkan GCN → batchs → tensor-split → ordre GPU → quants

### Campagne B — GPT-OSS 120B

- Réf. figée : S-23, `-t 8`, `--moe-cache off`
- Axes : NUMA / affinité / placement experts ; Vulkan GCN **à part**

Ne pas mélanger les CSV des deux campagnes.

---

## Livrables dépôt

| Élément | Rôle |
| ------- | ---- |
| `patches/gcn-fa-lds.patch` | build A |
| `patches/gcn-mmq-proprietary.patch` | build B |
| `patches/gcn-fa-mmq-combined.patch` | build A+B |
| `scripts/appliquer_patch_gcn.ps1` | appliquer / vérifier |
| `scripts/benchmark_polaris.ps1` | PP/TG, warm-up, ≥5 runs, CSV, métadonnées build |
| `scripts/lancer_70b_6gpu.ps1` | `-Batch`, `-UBatch`, `-Threads`, `-ThreadsBatch`, `-TensorSplit` |

---

## Hiérarchie de travail

| # | Expérience | Objectif | Statut |
| - | ---------- | -------- | ------ |
| 1 | Patch Vulkan GCN (4 builds) | PP | **Fait** — MMQ |
| 2 | `-b` / `-ub` | PP | **Fait** — `1024:256` |
| 3 | `--tensor-split` | équilibre GPU / VRAM | à faire |
| 4 | Ordre PCIe | transferts | à faire |
| 5 | NUMA (GPT-OSS) | experts CPU | à faire |
| 6 | Quants GGUF | TG + VRAM | à faire |
| 7 | Spéculation | TG | à faire |
| 8 | Fréquences GPU | bande passante | à faire |

**Verdict opérationnel :** travailler sur `builds/mmq` ; balayer `-b`/`-ub` puis éventuellement `-tb`, avant tensor-split.
