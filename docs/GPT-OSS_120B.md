# Projet expérimental : GPT-OSS 120B Q4_K_M sur 6× Polaris

> **Expérimental** — ne remplace pas la configuration [validée Llama 3.3 70B](../README.md). Même stack : Windows Vulkan, `llama-server`, 6× Polaris 8 Go.

Ce runbook décrit le scénario **GPT-OSS 120B** (MoE) sur les 6 GPU AMD Polaris, avec les résultats des tests locaux (2026-09-11).

## Caractéristiques

| Élément | Valeur |
|---------|--------|
| Modèle | GPT-OSS 120B |
| Architecture | MoE — 36 couches, **128 experts / couche**, top-4 actifs / token |
| Quantification | Q4_K_M (Unsloth) |
| Taille du GGUF | ~62,8 Go (2 shards) |
| Paramètres | ~117 B total / ~**5,1 B actifs** par token |
| GPU | 6× AMD Polaris 8 Go (48 Go VRAM) |
| RAM système | 128 Go |
| Backend | Vulkan |
| Runtime | `llama-server` |

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

| Config testée | Résultat |
|---------------|----------|
| Pas de `-ot` (tous experts GPU) | **OOM** dès le chargement (`ErrorOutOfDeviceMemory`, Vulkan0) |
| `--n-cpu-moe 30` | **OOM** (Vulkan5) |
| `--n-cpu-moe 32` | Charge, mais **seul Vulkan5** ~7 Go rempli — les autres presque vides |
| ~3 couches d’experts / GPU | **OOM** (buffers compute Vulkan0) |
| ~**2 couches d’experts / GPU** + reste CPU | **OK** — VRAM répartie |
| Tous experts CPU (`-ot ".ffn_.*_exps.=CPU"`) | **OK** — GPU quasi vides (~0,1–0,7 Go) |

Chaque couche d’experts pèse environ **1,6 Go**. Sur une Polaris 8 Go (hors compute / KV), le plafond réaliste est **~2–3 couches** d’experts résidentes.

## Stratégie recommandée : placement incrémental + cache MoE

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

Objectif runtime une fois le motif trouvé : couches résidentes + RAM + cache dynamique.

```text
                    GPT-OSS 120B Q4_K_M
                            │
              ┌─────────────┴─────────────┐
              │                           │
     Experts couches             Experts autres
     résidentes (incrémental)       couches (CPU)
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

`--moe-cache on` : budget `free-minus-reserve`. Ce n’est **pas** un placement permanent au chargement ; on ne peut pas non plus épingler à la main « l’expert n°17 » dans ce GGUF.

### Motif `-ot` issu du placement incrémental (2026-09-12)

Script : `.\scripts\placer_experts_incremental.ps1` (params + auto-detect devices/couches + option `-ValiderCtx`).

**Retenu pour ctx 8192** — asymétrique (Vulkan0 garde de la place pour ~1,1 Go de compute) :

| GPU | Couches experts résidentes | Buffer modèle |
|-----|----------------------------|---------------|
| Vulkan0 | **3** (`blk.0–2`) | ~4979 MiB |
| Vulkan1–4 | **4** | ~6577 MiB |
| Vulkan5 | **4** | ~7148 MiB |

```powershell
-ot "blk\.(3|4|5|6|11|12|17|18|23|24|29|30|35)\.ffn_.*_exps.=CPU"
```

Progression probe (ctx 2048) :

| Tour | Couches / carte | Buffer typique |
|------|-----------------|----------------|
| 1 | 1 | ~1,7 Go |
| 2 | 2 | ~3,3 Go |
| 3 | 3 | ~5,0 Go |
| 4 | 4 partout | ~6,6–7,1 Go — OK en probe ; **OOM compute Vulkan0** à ctx 8192 |
| 5 | 5 | **OOM** poids |

Donc : maximiser jusqu’à 4 couches, puis **reculer d’une couche sur Vulkan0** pour le contexte réel.

## Lancement recommandé

```powershell
$env:SERVEUR_LLAMA = "$env:USERPROFILE\Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
$env:MODELE_LLAMA  = "C:\chemin\vers\gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

& $env:SERVEUR_LLAMA `
  -m $env:MODELE_LLAMA `
  --host 127.0.0.1 --port 8080 `
  -c 8192 `
  -ngl 99 `
  -fa on `
  --split-mode layer `
  --device Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4,Vulkan5 `
  -ot "blk\.(3|4|5|6|11|12|17|18|23|24|29|30|35)\.ffn_.*_exps.=CPU" `
  --moe-cache on `
  --fit off `
  -v --log-timestamps
```

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

| Config | Prompt | Génération |
|--------|--------|------------|
| Tous experts CPU | ~0,51 tok/s | ~1,40 tok/s |
| `cpu4` (~2 couches / GPU) + moe-cache | ~0,86 tok/s | ~2,23 tok/s |
| **Incrémental asym. (3+4 couches) + moe-cache** | **~1,55 tok/s** | **~3,03 tok/s** |

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

Pour tenter une **5ᵉ couche** d’experts / GPU : OOM constaté au probe. Garder la marge compute (surtout Vulkan0 ~1,1 Go).

## Comparaison avec le 70B

| | Llama 3.3 70B Q4 | GPT-OSS 120B Q4 |
|--|------------------|-----------------|
| Statut | **Validé** | Expérimental |
| Nature | Dense | MoE |
| VRAM | Quasi pleine sur les 6 cartes | Partielle + RAM + cache MoE |
| ctx typique | 25600 | **32768** (TQ) / 8192 sans TQ |

Le 70B reste la config de référence pour la plateforme. GPT-OSS sert à évaluer le compromis MoE / Polaris / RAM.

## TurboQuant

Optionnel, **après** stabilité du chargement et de l’inférence de base.

TurboQuant compresse le **cache KV** (`--cache-type-k` / `--cache-type-v`).

Sur GPT-OSS, le KV est déjà petit (sliding window `n_swa=128`, moitié des couches SWA) à ctx modéré : le gain apparaît surtout en **montant le contexte**. Sur Polaris, le correctif `wave64` est décrit dans [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md).

### Test local (2026-09-12)

Config : motif incrémental asym. + `--moe-cache on` + `--cache-type-k q8_0 --cache-type-v turbo3` (binaire TheTom patché Polaris).

| | Charge + smoke | KV non-SWA | KV SWA | Total KV | Prompt | Génération |
|--|----------------|------------|--------|----------|--------|------------|
| Réf. f16 @ 8192 (`cpu4`) | OK | 288 MiB | 36 MiB | **~324 MiB** | ~0,86–1,75 | ~2,2 |
| **TQ @ 8192** | OK, pas d’assert wave64 | 132,75 MiB (K q8_0 76,5 + V turbo3 56,25) | 16,59 MiB | **~149 MiB** | **~1,69** | **~2,97** |
| **TQ @ 16384** | OK | 265,50 MiB (K 153 + V ~112,5) | 16,59 MiB | **~282 MiB** | **~1,26** | **~3,04** |
| **TQ @ 25600** | OK | 414,84 MiB (K 239 + V ~176) | 16,59 MiB | **~431 MiB** | **~1,54** | **~2,77** |
| **TQ @ 32768** | OK | 531,00 MiB (K 306 + V ~225) | 16,59 MiB | **~548 MiB** | **~1,51** | **~2,60** |

À ctx 8192 : KV ≈ **−54 %** vs f16 (≈ 175 MiB gagnés). Absolu encore petit face aux poids / experts. Progression **8192 → 32768** stable (pas d’OOM / pas d’assert) ; le SWA reste plafonné à 1024 cellules (≈ 16,6 MiB). Débits un peu plus bas à ctx élevé (overhead réservation / sched).

Lancement recommandé pour ctx long :

```powershell
# Même lancement recommandé (motif -ot + moe-cache), avec p.ex. -c 32768 et :
  --cache-type-k q8_0 --cache-type-v turbo3
```
