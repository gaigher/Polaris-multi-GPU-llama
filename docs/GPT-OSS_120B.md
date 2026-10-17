# Projet expérimental : GPT-OSS 120B Q4_K_M sur 6× Polaris

> **Expérimental** — ne remplace pas la configuration [validée Llama 3.3 70B](../README.md). Même stack : Windows Vulkan, `llama-server`, 6× Polaris 8 Go.
>
> Ce document distingue clairement :
>
> - ce qui est **mesuré / retenu** comme référence expérimentale ;
> - ce qui reste une **piste de test** (pas encore départagé).

Ce runbook décrit le scénario **GPT-OSS 120B** (MoE) sur les 6 GPU AMD Polaris, avec les résultats des tests locaux (2026-09-11 / 2026-09-12).

## Configuration expérimentale optimale actuelle

Référence à utiliser en priorité pour reproduire et comparer. Tout le reste du fichier documente *comment* on y est arrivé et *quoi tester ensuite*.

```text
Configuration expérimentale optimale actuelle
----------------------------------------------
Modèle       : GPT-OSS 120B Q4_K_M
GPU          : 6× Polaris 8 Go
Placement    : strat-23 (S) — 3+4+4+4+4+4, résidents en fin de bloc
             : bat A : ~+13 % gen @ 8k / ~+9 % @ 32k
Contexte     : 32768
KV K         : q8_0
KV V         : turbo3
MoE cache    : **off** (obligatoire en prod Polaris 8 Go)
             : 512 alloue OK après patch [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401),
               mais ~9× plus lent (thrash) — ne pas activer
Threads (-t) : 8   (validé @8k et @32k)
Flash Attn   : on
Backend      : Vulkan
```


| Métrique   | Valeur observée (S, 3 runs) |
| ---------- | --------------------------- |
| Prompt 8k  | ~3,14 tok/s (`-t 8`) / ~2,73 défaut |
| Gen 8k     | ~3,94 tok/s (`-t 8`) / ~3,35 défaut |
| Prompt 32k | ~3,03 tok/s (`-t 8`) / ~2,66 hist. |
| Gen 32k    | ~3,53 tok/s (`-t 8`) / ~3,16 hist. |
| KV @ 32k   | ~548 MiB                    |


**Résultat expérimental principal** (placement A historique, ctx 8192, sans TurboQuant) :


| Baseline                               | → Config incrémentale                             | Gain       |
| -------------------------------------- | ------------------------------------------------- | ---------- |
| Tous experts CPU : **~1,40 tok/s** gen | **3+4+4+4+4+4** + moe-cache : **~3,03 tok/s** gen | **≈ ×2,2** |
| ~0,51 tok/s prompt                     | ~1,55 tok/s prompt                                | **≈ ×3**   |


Bench position (2026-09-27) : à budgets VRAM égaux, **quelles** couches résidentes comptent. Profil **S (fin de bloc)** bat **A (début de bloc)** à 8k et 32k. Voir [§2](#2-même-total-23--la-position-des-couches-compte-t-elle-).

## Caractéristiques


| Élément        | Valeur                                                           |
| -------------- | ---------------------------------------------------------------- |
| Modèle         | GPT-OSS 120B                                                     |
| Architecture   | MoE — 36 couches, **128 experts / couche**, top-4 actifs / token |
| Quantification | Q4_K_M (Unsloth)                                                 |
| Taille du GGUF | ~62,8 Go (2 shards)                                              |
| Paramètres     | ~117 B total / ~**5,1 B actifs** par token                       |
| GPU            | 6× AMD Polaris 8 Go (48 Go VRAM)                                 |
| RAM système    | 128 Go                                                           |
| Backend        | Vulkan                                                           |
| Runtime        | `llama-server`                                                   |


Les « experts » sont des **MLP FFN SwiGLU anonymes** (pas des spécialistes nommés maths/code/…). Le routeur choisit 4 experts par token et par couche. Dans le GGUF, les 128 experts d’une couche sont **emballés dans les mêmes tenseurs** (`ffn_*_exps`) : `-ot` place donc une **couche entière**, pas un expert isolé.

Le fichier `Q4_K_M` est réparti en deux GGUF. **Les deux fichiers sont nécessaires** :

```text
gpt-oss-120b-Q4_K_M-00001-of-00002.gguf   (~49,6 Go)
gpt-oss-120b-Q4_K_M-00002-of-00002.gguf   (~13,1 Go)
```



## Télécharger le modèle

Dépôt : [unsloth/gpt-oss-120b-GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF). Les poids ne font pas partie de ce dépôt ; respecter la licence (Apache-2.0).

Exemple de répertoire local (les deux shards dans le même dossier) :

```text
C:\chemin\vers\gpt-oss-120b-GGUF\Q4_K_M\
```



## Pourquoi tout ne tient pas en VRAM

~58 Go de poids pour **48 Go** de VRAM : le Q4_K_M **entier** sur GPU est impossible.


| Config testée                                | Résultat                                                              |
| -------------------------------------------- | --------------------------------------------------------------------- |
| Pas de `-ot` (tous experts GPU)              | **OOM** dès le chargement (`ErrorOutOfDeviceMemory`, Vulkan0)         |
| `--n-cpu-moe 30`                             | **OOM** (Vulkan5)                                                     |
| `--n-cpu-moe 32`                             | Charge, mais **seul Vulkan5** ~7 Go rempli — les autres presque vides |
| ~3 couches d’experts / GPU (symétrique naïf) | **OOM** (buffers compute Vulkan0)                                     |
| ~**2 couches d’experts / GPU** + reste CPU   | **OK** — VRAM répartie                                                |
| **3+4+4+4+4+4** (incrémental)                | **OK** — référence expérimentale actuelle                             |
| 5 couches / GPU                              | **OOM** poids (probe)                                                 |
| Tous experts CPU (`-ot ".ffn_.*_exps.=CPU"`) | **OK** — GPU quasi vides (~0,1–0,7 Go)                                |


Chaque couche d’experts pèse environ **1,6 Go**. Sur une Polaris 8 Go (hors compute / KV), le plafond réaliste est **~2–4 couches** d’experts résidentes — **4** sur la plupart des cartes, **3** sur Vulkan0 (marge compute ~1,1 Go).

## Stratégie : deux niveaux (statique + cache)



### Niveau 1 — placement incrémental (résident)

Ne pas sauter directement à « N couches / GPU ». Procédure validée conceptuellement :

1. Partir de **tous les experts en CPU**.
2. **Tour 1** : résider **1 couche d’experts** sur chaque GPU (ex. `blk.0`→Vulkan0, `blk.7`→Vulkan1, …) ; recharger ; lire les `model buffer size` dans les logs.
3. **Tour suivant** : uniquement sur les GPU encore sous le seuil VRAM, ajouter **une** couche de plus ; ne plus toucher aux GPU proches de saturation.
4. En cas d’OOM : annuler le tour pour le GPU fautif, le marquer plein, continuer sur les autres.
5. À la fin : `--moe-cache on` pour utiliser la VRAM libre restante avec les experts CPU chauds.

Script automatisant cette boucle (reload à chaque tour — llama.cpp ne déplace pas à chaud) :

```powershell
# Requis : $env:SERVEUR_LLAMA, $env:MODELE_LLAMA (shard 00001 si split)
.\scripts\placer_experts_incremental.ps1 -ValiderCtx 8192 -MoeCache
# Auto-detect devices + carte couches ; seuils = % VRAM - marge compute
# -Json pour un resultat machine-readable a cote du .txt
```

> **EXPERIMENTAL** — best-effort, dépendant du modèle / backend / VRAM libre. Pas une API stable.

Le script actuel raisonne surtout en « combien de couches puis-je mettre ? ». Évolution souhaitée : raisonner en **VRAM volontairement libre pour le cache** (poids max → réserve compute → réserve KV → reste cache), avec une pénalité si la VRAM restante est trop faible — pas seulement `MAX couches`.

### Niveau 2 — cache dynamique (`--moe-cache`)

Deux chemins distincts :

```text
poids experts GPU  →  calcul GPU          (résident permanent)
poids experts CPU  →  RAM  →  moe-cache  →  VRAM temporaire  →  calcul GPU
```

`--moe-cache on` : budget `free-minus-reserve`. Ce n’est **pas** un placement permanent au chargement ; on ne peut pas non plus épingler à la main « l’expert n°17 » dans ce GGUF.

Ne **pas** viser 8,0 / 8,0 Go de VRAM. Vue utile d’une Polaris :

```text
8 Go
├── poids résidents
├── buffers compute   (~1,1 Go de marge observée sur Vulkan0 @ 8192)
├── KV                (~548 MiB @ 32k avec K=q8_0, V=turbo3)
└── espace libre  →  moe-cache
```

Objectif runtime : **GPU permanent + GPU cache dynamique + RAM**, pas seulement « max couches résidentes ».

```text
                    GPT-OSS 120B Q4_K_M
                            │
              ┌─────────────┴─────────────┐
              │                           │
     Experts couches             Experts autres
     résidentes (statique)           couches (CPU)
              │                           │
       6× Polaris 8 Go                  RAM 128 Go
              │                           │
              │              --moe-cache on
              │         (experts CPU chauds →
              │          VRAM libre, dynamique)
              └─────────────┬─────────────┘
                            │
                       Inférence MoE
```

À **32k**, priorité VRAM proposée : (1) buffers compute, (2) KV, (3) moe-cache, (4) seulement ensuite une couche résidente de plus. Préférer **22 couches + ~1–2 Go de marge/cache** à **23 couches + VRAM saturée**, *si* les benches confirment que le cache compense.

### Placement de référence expérimentale : strat-23 (S)

**Retenu** (2026-09-27) — mêmes budgets **3+4+4+4+4+4** que A, mais experts résidents en **fin de chaque bloc** GPU (Vulkan0 inchangé `0–2` pour la marge compute).


| GPU       | Couches experts résidentes (S) | Buffer modèle |
| --------- | ------------------------------ | ------------- |
| Vulkan0   | **3** (`blk.0–2`)              | ~4979 MiB     |
| Vulkan1   | **4** (`blk.9–12`)             | ~6577 MiB     |
| Vulkan2   | **4** (`blk.15–18`)            | ~6577 MiB     |
| Vulkan3   | **4** (`blk.21–24`)            | ~6577 MiB     |
| Vulkan4   | **4** (`blk.27–30`)            | ~6577 MiB     |
| Vulkan5   | **4** (`blk.32–35`)            | ~7148 MiB     |
| **Total** | **23 / 36**                    | —             |


```powershell
# S (réf. actuelle)
-ot "blk\.(3|4|5|6|7|8|13|14|19|20|25|26|31)\.ffn_.*_exps.=CPU"
# A (historique, début de bloc) — plus lent
# -ot "blk\.(3|4|5|6|11|12|17|18|23|24|29|30|35)\.ffn_.*_exps.=CPU"
```

Progression probe (ctx 2048) :


| Tour | Couches / carte | Buffer typique                                                 |
| ---- | --------------- | -------------------------------------------------------------- |
| 1    | 1               | ~1,7 Go                                                        |
| 2    | 2               | ~3,3 Go                                                        |
| 3    | 3               | ~5,0 Go                                                        |
| 4    | 4 partout       | ~6,6–7,1 Go — OK en probe ; **OOM compute Vulkan0** à ctx 8192 |
| 5    | 5               | **OOM** poids                                                  |


Donc : maximiser jusqu’à 4 couches, puis **reculer d’une couche sur Vulkan0** pour le contexte réel. Ne pas chercher à forcer 5 couches / carte.

## Deux configurations de référence



### A — Court contexte (8192), débit max mesuré

Sans TurboQuant. Placement **S** (strat-23), **`-t 8`**, `--moe-cache off`.


| Prompt (avg 3) | Génération (avg 3) |
| -------------- | ------------------ |
| ~3,14 tok/s    | **~3,94 tok/s**    |


```powershell
$env:SERVEUR_LLAMA = "$env:USERPROFILE\Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
$env:MODELE_LLAMA  = "C:\chemin\vers\gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

& $env:SERVEUR_LLAMA `
  -m $env:MODELE_LLAMA `
  --host 127.0.0.1 --port 8080 `
  -c 8192 `
  -t 8 `
  -ngl 99 `
  -fa on `
  --split-mode layer `
  --device Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4,Vulkan5 `
  -ot "blk\.(3|4|5|6|7|8|13|14|19|20|25|26|31)\.ffn_.*_exps.=CPU" `
  --moe-cache off `
  --fit off `
  -v --log-timestamps
```



### B — Long contexte (32k) + TurboQuant

Même base que A (`-t 8`, moe-cache off, `-ot` S), avec :

```powershell
  -c 32768 `
  --cache-type-k q8_0 `
  --cache-type-v turbo3
```


| Prompt (avg 3) | Génération (avg 3) | KV       |
| -------------- | ------------------ | -------- |
| ~3,03 tok/s    | **~3,53 tok/s**    | ~548 MiB |


Le fichier `00001-of-00002.gguf` est le point d’entrée ; `llama.cpp` détecte le second shard dans le même répertoire.

Chat / API : [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

### Variante simple (tous experts CPU)

Utile pour valider le chargement uniquement ; GPU peu utilisés, plus lent :

```powershell
-ot ".ffn_.*_exps.=CPU"
# sans --moe-cache, ou avec si de la VRAM reste libre
```



## Performances mesurées

Plateforme de test : 1× RX 570 + 5× RX 580, Xeon E5-2620 v3, 128 Go RAM, build `llama-server` Vulkan local.


| Config                                | Prompt          | Génération      | Statut        |
| ------------------------------------- | --------------- | --------------- | ------------- |
| Tous experts CPU                      | ~0,51 tok/s     | ~1,40 tok/s     | Baseline      |
| `cpu4` (~2 couches / GPU) + moe-cache | ~0,86 tok/s     | ~2,23 tok/s     | Intermédiaire |
| A début de bloc + moe-cache @ 8192    | ~2,55 tok/s     | ~2,94 tok/s     | Ancienne réf. |
| **S fin de bloc + moe-cache @ 8192**  | **~2,78 tok/s** | **~3,40 tok/s** | **Réf. 8k**   |
| A + TQ @ 32768                        | ~2,28 tok/s     | ~2,64 tok/s     | Ancienne réf. |
| **S + TQ @ 32768**                    | **~2,66 tok/s** | **~3,16 tok/s** | **Réf. 32k**  |


Le goulot principal reste le **CPU** pour les experts non résidents / non en cache. Avec le motif incrémental, la VRAM poids approche ~6,6–7,1 Go / carte (probe), bien plus proche du 70B dense.

## Vérifications

Après le lancement, contrôler dans les logs :

- six périphériques Vulkan listés et couches réparties ;
- `MoE cache requested=on resolved=on` ;
- buffers modèle par GPU (ordre de grandeur ci-dessus) ;
- absence d’`ErrorOutOfDeviceMemory` / `GGML_ASSERT` ;
- débits prompt et génération.

Progression de contexte (validée avec TurboQuant, 2026-09-12) :

```text
8192 → 16384 → 25600 → 32768   (charge + smoke OK à chaque palier)
```



## Comparaison avec le 70B


|             | Llama 3.3 70B Q4              | GPT-OSS 120B Q4                             |
| ----------- | ----------------------------- | ------------------------------------------- |
| Statut      | **Validé**                    | Expérimental                                |
| Nature      | Dense                         | MoE                                         |
| VRAM        | Quasi pleine sur les 6 cartes | Partielle + RAM + cache MoE                 |
| Placement   | Dense layer-split             | **23 experts GPU (S)** + 13 CPU + moe-cache |
| ctx typique | 25600                         | **32768** (TQ) / 8192 sans TQ               |
| Gen typique | (voir README 70B)             | ~3,34 @ 8k / ~2,87 @ 32k TQ (S, 3 runs)     |


Le 70B reste la config de référence **validée** pour la plateforme. GPT-OSS sert à évaluer le compromis MoE / Polaris / RAM.

## TurboQuant

Optionnel pour le débit court ; **recommandé** pour la config B (ctx long).

TurboQuant compresse le **cache KV** (`--cache-type-k` / `--cache-type-v`).

Sur GPT-OSS, le KV est déjà petit (sliding window `n_swa=128`, moitié des couches SWA) à ctx modéré : le gain apparaît surtout en **montant le contexte**. Sur Polaris, le correctif `wave64` est décrit dans [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md).

### Matrice ctx (2026-09-12)

Config : motif **3+4+4+4+4+4** + `--moe-cache on` + `--cache-type-k q8_0 --cache-type-v turbo3` (binaire TheTom patché Polaris).


| ctx                      | Charge + smoke          | KV non-SWA | KV SWA    | Total KV     | Prompt     | Génération |
| ------------------------ | ----------------------- | ---------- | --------- | ------------ | ---------- | ---------- |
| Réf. f16 @ 8192 (`cpu4`) | OK                      | 288 MiB    | 36 MiB    | **~324 MiB** | ~0,86–1,75 | ~2,2       |
| **TQ @ 8192**            | OK, pas d’assert wave64 | 132,75 MiB | 16,59 MiB | **~149 MiB** | **~1,69**  | **~2,97**  |
| **TQ @ 16384**           | OK                      | 265,50 MiB | 16,59 MiB | **~282 MiB** | **~1,26**  | **~3,04**  |
| **TQ @ 25600**           | OK                      | 414,84 MiB | 16,59 MiB | **~431 MiB** | **~1,54**  | **~2,77**  |
| **TQ @ 32768**           | OK                      | 531,00 MiB | 16,59 MiB | **~548 MiB** | **~1,51**  | **~2,60**  |


À ctx 8192 : KV ≈ **−54 %** vs f16 (≈ 175 MiB gagnés). Absolu encore petit face aux poids / experts. Progression **8192 → 32768** stable (pas d’OOM / pas d’assert) ; le SWA reste plafonné à 1024 cellules (≈ 16,6 MiB). Débits un peu plus bas à ctx élevé (overhead réservation / sched).

---



## Pistes de test prioritaires

Ne pas modifier plusieurs paramètres à la fois (`-fa`, TurboQuant, CPU, etc. restent fixes) sauf indication contraire.

### 0. Résultat bench A/B/C (2026-09-27) — **départagé**

Script : `scripts/bench_moe_cache_abc.ps1` (warmup + mesure, même prompt, `--moe-cache on`).

`-ot` **utilisés :**

```text
A (23) : blk\.(3|4|5|6|11|12|17|18|23|24|29|30|35)\.ffn_.*_exps.=CPU
B (22) : + blk.10 en CPU   → 3+3+4+4+4+4
C (21) : + blk.10 et 16    → 3+3+3+4+4+4
```



#### ctx 8192 (sans TurboQuant)


| Profil | Motif       | Couches | VRAM typique (MiB)                     | Prompt*  | Gen*     |
| ------ | ----------- | ------- | -------------------------------------- | -------- | -------- |
| **A**  | 3+4+4+4+4+4 | 23      | 4979 / 6577×4 / 7148                   | **3,08** | **3,66** |
| B      | 3+3+4+4+4+4 | 22      | 4979 / **4959** / 6577×3 / 7148        | 2,27     | 2,87     |
| C      | 3+3+3+4+4+4 | 21      | 4979 / 4959 / **4959** / 6577×2 / 7148 | 2,55     | 3,03     |


2ᵉ requête (après warmup). Warm A : ~1,53 prompt / ~3,50 gen.

**Verdict @ 8192 :** libérer ~1,6–3,2 Go pour le cache **pénalise** nettement. **A (23 couches) reste optimal** pour le débit court.

#### ctx 32768 (K=q8_0, V=turbo3) — confirmation 3 runs (2026-09-27)


| Profil | Motif       | Couches | Prompt avg | Gen avg  | Gen min–max |
| ------ | ----------- | ------- | ---------- | -------- | ----------- |
| **A**  | 3+4+4+4+4+4 | 23      | 2,18       | **2,55** | 2,54–2,56   |
| B      | 3+3+4+4+4+4 | 22      | 2,25       | 2,53     | 2,51–2,55   |


Run unique antérieur (B 2,94 > A 2,89) **non confirmé** : sur 3 mesures stables, **A reste légèrement devant en gen** (~+0,9 %). B a un prompt un peu meilleur. Écarts dans le bruit / charge machine (débits absolus plus bas que le run unique).

**Verdict @ 32768 :** garder **A (23 couches)** comme référence. Retirer une couche pour le cache n’apporte pas de gain gen reproductible.

CSV : `bench_results/abc-20260927-173801.csv` (8k), `abc-20260927-175826.csv` (32k 1-run), `abc-20260927-182043.csv` (32k 3-runs A/B).

### 1. Matrice couches × contexte (remplie)


| Profil | Couches GPU | Cache | ctx   | Prompt | Gen  | Statut                     |
| ------ | ----------- | ----- | ----- | ------ | ---- | -------------------------- |
| A      | 23          | ON    | 8192  | 3,08   | 3,66 | **meilleur @ 8k**          |
| B      | 22          | ON    | 8192  | 2,27   | 2,87 | plus lent                  |
| C      | 21          | ON    | 8192  | 2,55   | 3,03 | plus lent                  |
| **A**  | 23          | ON    | 32768 | 2,18   | 2,55 | **rétabli @ 32k (3 runs)** |
| B      | 22          | ON    | 32768 | 2,25   | 2,53 | pas de gain gen confirmé   |
| C      | 21          | ON    | 32768 | 2,55   | 2,92 | 1 run seulement            |




### 2. Même total 23 — la *position* des couches compte-t-elle ?

Hypothèse : toutes les couches n’ont pas la même valeur. Tester sans changer le total **3+4+4+4+4+4**.

**Contrainte importante :** avec `--split-mode layer`, la carte couche→GPU est fixe (`V0:0–6 … V5:31–35`). `-ot` ne fait que forcer des experts en CPU : on **ne peut pas** « envoyer la couche 3 sur Vulkan1 ». Un motif « stratifié cross-GPU » du type `GPU1: 3 4 5 6` est donc **invalide** tel quel (ces experts resteraient sur Vulkan0 → OOM).

Profil **S** testé = même budgets par carte, mais résidents en **fin de bloc** (sauf Vulkan0 gardé `0,1,2`) :

```text
# A — début de bloc (réf.)
V0: 0 1 2 | V1: 7 8 9 10 | V2: 13–16 | V3: 19–22 | V4: 25–28 | V5: 31–34
CPU: 3 4 5 6 11 12 17 18 23 24 29 30 35

# S — fin de bloc (stratifié intra-GPU)
V0: 0 1 2 | V1: 9 10 11 12 | V2: 15–18 | V3: 21–24 | V4: 27–30 | V5: 32–35
CPU: 3 4 5 6 7 8 13 14 19 20 25 26 31
-ot S: blk\.(3|4|5|6|7|8|13|14|19|20|25|26|31)\.ffn_.*_exps.=CPU
```

Bench A vs S en cours (`-ProfileFilter A,S -Runs 3 -CtxMode both`).

**Résultat (2026-09-27, 3 runs)** — CSV `bench_results/abc-20260927-184928.csv` :


| Placement | Ctx   | Prompt avg | Gen avg  | Gen min–max | vs A gen  |
| --------- | ----- | ---------- | -------- | ----------- | --------- |
| A début   | 8192  | 2,55       | 2,94     | 2,91–2,97   | —         |
| **S fin** | 8192  | 2,74       | **3,34** | 3,33–3,34   | **+13 %** |
| A début   | 32768 | 2,28       | 2,64     | 2,63–2,66   | —         |
| **S fin** | 32768 | 2,44       | **2,87** | 2,87–2,88   | **+9 %**  |


**Verdict :** à VRAM égale, **la position des couches résidentes compte**. S devient la référence. Buffers modèle identiques à A (même nombre de couches / GPU).


| Placement      | Prompt tok/s | Gen tok/s | Notes                       |
| -------------- | ------------ | --------- | --------------------------- |
| A 3+4+4+4+4+4  | 2,55         | 2,94      | début de bloc @ 8192        |
| **S strat-23** | **2,74**     | **3,34**  | fin de bloc — **rétention** |
| 4+4+4+4+4+3    | ?            | ?         | Vulkan0 à 4 → risque OOM    |




### 3. Variants de S + stats moe-cache (2026-09-27)



#### Throughput S / M / H (3 runs)

Budgets identiques **3+4+4+4+4+4**, Vulkan0=`0,1,2`. CSV : `bench_results/abc-20260927-201607.csv`.


| Profil | Motif            | Ctx   | Prompt avg | Gen avg  | vs S gen |
| ------ | ---------------- | ----- | ---------- | -------- | -------- |
| **S**  | fin de bloc      | 8192  | 2,78       | **3,40** | —        |
| H      | 1 early + 3 late | 8192  | 2,59       | 3,38     | −0,7 %   |
| M      | milieu de bloc   | 8192  | 2,60       | 3,13     | −7,8 %   |
| **S**  | fin de bloc      | 32768 | 2,66       | **3,16** | —        |
| H      | 1 early + 3 late | 32768 | 2,36       | 2,55     | −19 %    |
| M      | milieu de bloc   | 32768 | 2,19       | 2,52     | −20 %    |


**Verdict :** **S (fin de bloc) reste optimal**. H ≈ S à 8k mais s’effondre à 32k. M est systématiquement plus lent. Confirme le biais « couches tardives du bloc → GPU ».

#### Stats hits/misses (probe `-lv 4`, 2026-09-27)


| Mode                                 | Résultat observé                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--moe-cache on` (défaut S)          | `resolved=on` mais **aucune** ligne `[moe-cache]` pool/hits → cache **inactif** en pratique (budget free−reserve ≈ 0 sur Polaris 8 Go déjà remplies) |
| `--moe-cache 512` + `RESERVE_MB=256` | `[moe-cache] session ready` puis `failed to allocate 508 MiB expert pool` en boucle                                                                  |
| hits=                                | **jamais observé** sur cette ferme avec 23 couches résidentes                                                                                        |


Conséquence : les gains A→S (et les pertes M/H) viennent du **placement statique** `-ot`, pas du cache dynamique. `--moe-cache on` dans la config S actuelle ne remplit pas de pools VRAM.

Pour qu’un vrai cache existe, il faudrait d’abord **libérer assez de VRAM** (moins de couches résidentes) *et* vérifier `pool[` + `hits=` sous `-lv 4` — puis seulement comparer le débit à S.

#### Probe S-22 (V5 `4→3`, retire `blk.35`) — alloc cache (2026-09-27)

Même fin de bloc que S, une seule couche en moins sur Vulkan5 (`buffer V5` 7148→5530 MiB). Script : `scripts/probe_moe_cache_alloc.ps1`. CSV : `bench_results/moe-alloc-S22-20260927-213637.csv`.


| Budget MiB | resolved   | session ready | pools | fail_alloc | hits |
| ---------- | ---------- | ------------- | ----- | ---------- | ---- |
| 256        | on→**off** | non           | 0     | 0          | 0    |
| 512        | on→on      | oui           | 0     | 5502       | 0    |
| 768        | on→on      | oui           | 0     | 5502       | 0    |
| 1024       | on→on      | oui           | 0     | 5502       | 0    |


`configured: devices=1` puis `failed to allocate 508 MiB expert pool`. **Libérer V5 ne suffit pas** — et pour cause (voir ci-dessous).

#### Device de la session cache = **Vulkan0 uniquement**

Dans `ggml-vulkan-moe-cache.cpp` : `min_devices = 1`, premier backend Vulkan du scheduler → **Vulkan0**. V5 n’entre jamais dans la session.

#### Probes ciblés Vulkan0 / S-21 (2026-10-03)


| Profil                | V0 poids     | V5 poids | Budget  | session | pools | fail_alloc | hits |
| --------------------- | ------------ | -------- | ------- | ------- | ----- | ---------- | ---- |
| S22V0 (V0=`0,1`)      | 3361 MiB     | 7148     | 512–768 | oui     | 0     | ~5500      | 0    |
| S21 (V0=`0,1` + V5−1) | 3361 MiB     | 5530     | 512–768 | oui     | 0     | ~5900      | 0    |
| S20V0 (V0=`0` seul)   | **1743 MiB** | 7148     | 512     | oui     | 0     | ~5900      | 0    |
| S20V0                 | 1743 MiB     | 7148     | 64–256  | **off** | 0     | 0          | 0    |


Même avec **~6 Go libres** sur Vulkan0 (S20V0), toujours `failed to allocate 508 MiB` en mode `host-mapped`. Budgets < ~512 → `no cacheable expert tensors` (slab+scratch > budget).

**Cause (confirmée 2026-10-04) :** le backend Vulkan détectait un type `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (aperture CPU ~256 MiB / faux UMA AMD) et choisissait host-mapped ; sur Polaris discret ce tas est trop petit → l’alloc échouait. Le chemin GPU discret (`device-local` + staging) n’était **pas** pris.

#### Patch Vulkan + re-probe S-23 (2026-10-04)

Patch `ggml-vulkan-moe-cache.cpp` : host-mapped seulement si tas UMA ≥ budget ; sinon device-local + staging ; override `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1`.  
PR upstream : [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) — runbook [PR_MOE_CACHE_VULKAN.md](PR_MOE_CACHE_VULKAN.md).

**Alloc (probe S,** `--moe-cache 512`**,** `RESERVE_MB=128`**) — OK :**

```text
[moe-cache] Vulkan: UMA heap=256 MiB < need=512 MiB; using device-local + staging (discrete/aperture path)
[moe-cache] Vulkan session ready (budget=512 MiB, device-local)
[moe-cache] Vulkan0 pool[0]: type=mxfp4 expert=4303 KiB slots=121 ... total=508 MiB
[moe-cache] enabled: first pool allocated on Vulkan0
```


| Probe                   | session | pools | fail_alloc | hits fin |
| ----------------------- | ------- | ----- | ---------- | -------- |
| S-23 @512 (avant patch) | oui     | 0     | ≫0         | 0        |
| S-23 @512 (après)       | oui     | **1** | **0**      | ~49 %    |


**Débit S-23 @8192 ×3** (`bench_results/abc-20261004-172308`) :


| moe-cache | PromptAvg | GenAvg    | notes                                                     |
| --------- | --------- | --------- | --------------------------------------------------------- |
| **off**   | **2,725** | **3,352** | réf. prod                                                 |
| **512**   | 0,353     | **0,369** | ~9× plus lent ; hits 49,8 % mais thrashing (pool 121/121) |


**Verdict :** le patch **corrige l’alloc** ; sur Polaris 8 Go avec S-23, `--moe-cache 512` **régresse** fortement (staging + évictions). Prod : `--moe-cache off`, réf. = S-23.

```text
S fin   : V1 9–12,  V2 15–18, V3 21–24, V4 27–30, V5 32–35
… 
```



### 4. Suite utile (priorisée) — CPU d’abord, cache en dernier

Avec S-23 + `--moe-cache off` comme plafond GPU actuel, le goulot restant = **13 couches experts sur CPU** (deux Xeon). Ordre recommandé :

#### P1 — CPU / NUMA / threads (plus gros levier probable)

| # | Test | Comment | Succès si |
| - | ---- | ------- | --------- |
| 1 | Baseline S-23 `@8192`/`@32768`, `-t` défaut | `bench_moe_cache_abc.ps1 -ProfileFilter S -MoeCache off -Runs 3` | chiffres stables |
| 2 | Varier `-t` (ex. 8 / 12 / 16 / 24) | même `-ot` S, un seul changement | gen ↑ sans régression prompt |
| 3 | Affinité NUMA | lancer `llama-server` piné nœud proche RAM/GPU si applicable ; noter `Get-Process` CPU | moins de cross-NUMA |
| 4 | Pendant gen | PerfMon / Task Manager : %CPU, fréquences, débit RAM | confirmer satur. CPU vs idle GPU |

Ne pas toucher `-fa`, TQ, `-ot` pendant P1.

**Sweep `-t` @8192 ×3** (`abc-20261004-180910`, S-23, moe-cache=off) — **terminé** :

| `-t` | PromptAvg | GenAvg | vs meilleur |
| ---: | --------: | -----: | ----------- |
| **8** | **3,136** | **3,940** | réf. |
| 12 | 3,100 | 3,651 | −7 % gen |
| 16 | 2,390 | 2,848 | −28 % gen |
| 24 | 2,189 | 3,001 | −24 % gen |

→ **retenir `-t 8`** (aussi meilleur prompt). Trop de threads nuit (contention / oversubscribe).

**Validation `-t 8` @32768 ×3** (`abc-20261004-183252`, S-23 off, TQ KV) — **OK** :

| Ctx | PromptAvg | GenAvg | vs hist. S (~défaut `-t`) |
| --: | --------: | -----: | ------------------------- |
| 8192 | 3,136 | 3,940 | gen ~+18 % |
| **32768** | **3,031** | **3,530** | gen ~+12 % (hist. ~3,16) |

`-t 8` validé aux deux contextes. Suite P1 optionnelle : NUMA ; sinon P2.

#### P2 — Variants `-ot` à **23** couches seulement

Déjà départagés : A vs S (S gagne), M/H (perdent), B/C (moins de couches).

**Sweep V0 @8192 ×3** (`abc-20261004-190642`, `-t 8`, moe-cache=off) — **terminé** :

| Profil | V0 GPU | PromptAvg | GenAvg | vs S |
| ------ | ------ | --------: | -----: | ---- |
| **S** | 0,1,2 | 3,212 | 3,976 | réf. |
| **SV0L** | 4,5,6 | 3,202 | **4,001** | +0,6 % (bruit) |
| SV013 | 0,1,3 | 3,177 | 3,800 | −4,4 % |
| SV023 | 0,2,3 | 3,199 | 3,651 | −8,2 % |

→ **garder S** (V0=`0,1,2`). SV0L ≈ S, pas assez pour changer la réf. sans confirm @32k. Frontières 013/023 **abandonnées**.

#### P3 — Cache seulement si marge V0 réelle (basse priorité)

Conditions **toutes** requises avant un nouveau bench cache :

1. Patch device-local présent (fait).
2. Assez de VRAM **device-local** libre sur Vulkan0 **après** poids+KV+compute — pas juste « buffer weight bas ».
3. Budget assez grand pour **réduire le thrash** (pool 121 slots @512 MiB ≈ 50 % hits + churn → échec perf).
4. Comparer à S-23 **off** : garder cache seulement si gen ≥ off.

Sinon laisser `--moe-cache off`.

### 5. Ce qu’il ne faut plus prioriser

- Forcer **5 couches** d’experts par Polaris → OOM déjà démontré.
- Retirer des couches pour « activer le cache » (V5, V0, S-21, S-20V0) → **infirmé** : le goulot était l’alloc host-mapped (maintenant corrigée) ; même avec pool OK, 512 MiB thrash vs S-23.
- Espérer un gain tok/s de `--moe-cache 512` sur S-23 Polaris 8 Go → **infirmé** (~9× plus lent).
- Espérer du multi-GPU moe-cache Vulkan → **1 device seulement** (Vulkan0).
- Re-bench A/M/H « pour voir » → déjà tranché.

### Objectif révisé

1. **Réf. débit = S-23** + **`-t 8`** + **`--moe-cache off`** (prod Polaris 8 Go — ne pas passer à `on`/`512`).
2. **moe-cache Vulkan** : correctness only — [PR #401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) / [PR_MOE_CACHE_VULKAN.md](PR_MOE_CACHE_VULKAN.md).
3. **Prochaine action** (optionnel) : confirm SV0L @32768 si on veut trancher le +0,6 % ; sinon P1-NUMA. P3 cache — pas urgent. Réf. reste **S + `-t 8` + moe-cache off**.

