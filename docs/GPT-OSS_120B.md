# Projet expérimental : GPT-OSS 120B Q4_K_M sur 6× Polaris

> **Expérimental** — ne remplace pas la configuration [validée Llama 3.3 70B](../README.md). Même stack : Windows Vulkan, `llama-server`, 6× Polaris 8 Go.
>
> **Optimisations MoE** (placement, `-t`, cache, HOT_USES) : tout est dans **[Plan Optimiser MoE.md](Plan%20Optimiser%20MoE.md)**. Ce fichier = runbook lancement + chiffres de référence.

Ce runbook décrit le scénario **GPT-OSS 120B** (MoE) sur les 6 GPU AMD Polaris.

## Configuration expérimentale optimale actuelle

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


| Métrique   | Valeur observée (S, 3 runs)         |
| ---------- | ----------------------------------- |
| Prompt 8k  | ~3,14 tok/s (`-t 8`) / ~2,73 défaut |
| Gen 8k     | ~3,94 tok/s (`-t 8`) / ~3,35 défaut |
| Prompt 32k | ~3,03 tok/s (`-t 8`) / ~2,66 hist.  |
| Gen 32k    | ~3,53 tok/s (`-t 8`) / ~3,16 hist.  |
| KV @ 32k   | ~548 MiB                            |


**Résultat expérimental principal** (placement A historique, ctx 8192, sans TurboQuant) :


| Baseline                               | → Config incrémentale                             | Gain       |
| -------------------------------------- | ------------------------------------------------- | ---------- |
| Tous experts CPU : **~1,40 tok/s** gen | **3+4+4+4+4+4** + moe-cache : **~3,03 tok/s** gen | **≈ ×2,2** |
| ~0,51 tok/s prompt                     | ~1,55 tok/s prompt                                | **≈ ×3**   |


Détail benches placement / cache / threads : [Plan Optimiser MoE.md](Plan%20Optimiser%20MoE.md).

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


Les « experts » sont des **MLP FFN SwiGLU anonymes**. Le routeur choisit 4 experts par token et par couche. Dans le GGUF, les 128 experts d’une couche sont **emballés** (`ffn_*_exps`) : `-ot` place une **couche entière**.

```text
F:\Models\gpt-oss-120b\Q4_K_M\
  gpt-oss-120b-Q4_K_M-00001-of-00002.gguf   (~46,2 Go)
  gpt-oss-120b-Q4_K_M-00002-of-00002.gguf   (~12,2 Go)
```

Les deux shards **doivent** rester dans le même dossier. Point d’entrée = `00001-of-00002.gguf`.

## Télécharger le modèle

Dépôt : [unsloth/gpt-oss-120b-GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF). Licence Apache-2.0. Copie locale recommandée : SSD `F:\Models\…` (`scripts\copier_gguf_vers_F.cmd`).

## Placement S-23 (rappel)


| GPU       | Couches experts (S)  | Buffer modèle |
| --------- | -------------------- | ------------- |
| Vulkan0   | **3** (`blk.0–2`)    | ~4979 MiB     |
| Vulkan1–4 | **4** (fins de bloc) | ~6577 MiB     |
| Vulkan5   | **4** (`blk.32–35`)  | ~7148 MiB     |
| **Total** | **23 / 36**          | —             |


```powershell
-ot "blk\.(3|4|5|6|7|8|13|14|19|20|25|26|31)\.ffn_.*_exps.=CPU"
```

Pourquoi 23 / fin de bloc / pas de cache : [Plan Optimiser MoE.md](Plan%20Optimiser%20MoE.md).

Script incrémental : `scripts/placer_experts_incremental.ps1`.

## Deux configurations de référence



### A — Court contexte (8192), débit max mesuré

Sans TurboQuant. Placement **S**, `-t 8`, `--moe-cache off`.


| Prompt (avg 3) | Génération (avg 3) |
| -------------- | ------------------ |
| ~3,14 tok/s    | **~3,94 tok/s**    |


```powershell
$env:SERVEUR_LLAMA = "$env:USERPROFILE\Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
$env:MODELE_LLAMA  = "F:\Models\gpt-oss-120b\Q4_K_M\gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

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

Même base que A, avec :

```powershell
  -c 32768 `
  --cache-type-k q8_0 `
  --cache-type-v turbo3
```


| Prompt (avg 3) | Génération (avg 3) | KV       |
| -------------- | ------------------ | -------- |
| ~3,03 tok/s    | **~3,53 tok/s**    | ~548 MiB |


Chat / API : [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

### Variante simple (tous experts CPU)

```powershell
-ot ".ffn_.*_exps.=CPU"
```



## Performances mesurées (historique)

Plateforme : 1× RX 570 + 5× RX 580, Xeon E5-2620 v3, 128 Go RAM.


| Config                                 | Prompt      | Génération  | Statut        |
| -------------------------------------- | ----------- | ----------- | ------------- |
| Tous experts CPU                       | ~0,51 tok/s | ~1,40 tok/s | Baseline      |
| `cpu4` (~2 couches / GPU) + moe-cache  | ~0,86 tok/s | ~2,23 tok/s | Intermédiaire |
| A début de bloc + moe-cache @ 8192     | ~2,55 tok/s | ~2,94 tok/s | Ancienne réf. |
| **S fin de bloc @ 8192** (`-t 8`, off) | **~3,14**   | **~3,94**   | **Réf. 8k**   |
| **S + TQ @ 32768** (`-t 8`, off)       | **~3,03**   | **~3,53**   | **Réf. 32k**  |




## Vérifications

- six périphériques Vulkan listés ;
- `--moe-cache off` (prod) — si `on` : logs `MoE cache` / pools (voir plan) ;
- buffers modèle par GPU (ordre de grandeur ci-dessus) ;
- absence d’`ErrorOutOfDeviceMemory` / `GGML_ASSERT` ;
- débits prompt et génération.

Progression ctx (TQ, 2026-09-12) : `8192 → 16384 → 25600 → 32768` OK.

## Comparaison avec le 70B


|             | Llama 3.3 70B Q4              | GPT-OSS 120B Q4                          |
| ----------- | ----------------------------- | ---------------------------------------- |
| Statut      | **Validé**                    | Expérimental                             |
| Nature      | Dense                         | MoE                                      |
| VRAM        | Quasi pleine sur les 6 cartes | Partielle + RAM (+ cache MoE si un jour) |
| Placement   | Dense layer-split             | **23 experts GPU (S)** + 13 CPU          |
| ctx typique | 25600                         | **32768** (TQ) / 8192 sans TQ            |
| Gen typique | (voir README 70B)             | ~3,94 @ 8k / ~3,53 @ 32k TQ (S, `-t 8`)  |




## TurboQuant

Optionnel pour le débit court ; **recommandé** pour la config B (ctx long). Correctif `wave64` : [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md).

### Matrice ctx (2026-09-12)

Config historique : motif **3+4+4+4+4+4** + `--moe-cache on` + `q8_0`/`turbo3` (avant décision prod `off`).


| ctx                      | Charge + smoke          | Total KV     | Prompt     | Génération |
| ------------------------ | ----------------------- | ------------ | ---------- | ---------- |
| Réf. f16 @ 8192 (`cpu4`) | OK                      | **~324 MiB** | ~0,86–1,75 | ~2,2       |
| **TQ @ 8192**            | OK, pas d’assert wave64 | **~149 MiB** | **~1,69**  | **~2,97**  |
| **TQ @ 16384**           | OK                      | **~282 MiB** | **~1,26**  | **~3,04**  |
| **TQ @ 25600**           | OK                      | **~431 MiB** | **~1,54**  | **~2,77**  |
| **TQ @ 32768**           | OK                      | **~548 MiB** | **~1,51**  | **~2,60**  |


À ctx 8192 : KV ≈ **−54 %** vs f16. Débits de la matrice = build / `-t` d’époque ; réf. actuelle = config B en tête de fichier.

## Optimisations MoE

Toutes les solutions (placement, `-t`, cache #401, HOT_USES) : **[Plan Optimiser MoE.md](Plan%20Optimiser%20MoE.md)**.
