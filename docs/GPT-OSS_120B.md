# Projet expérimental : GPT-OSS 120B Q4_K_M sur 6× Polaris

> **Expérimental** — ne remplace pas la configuration [validée Llama 3.3 70B](../README.md). Même stack : Windows Vulkan, `llama-server`, 6× Polaris 8 Go.

Ce runbook décrit un scénario pour **GPT-OSS 120B**, modèle **Mixture of Experts (MoE)**, sur les 6 GPU AMD Polaris.

## Caractéristiques

| Élément | Valeur |
|---------|--------|
| Modèle | GPT-OSS 120B |
| Architecture | MoE |
| Quantification | Q4_K_M |
| Taille du GGUF | ~62,8 Go (2 shards) |
| GPU | 6× AMD Polaris 8 Go |
| VRAM totale | 48 Go |
| RAM système | 128 Go |
| Backend | Vulkan |
| Runtime | `llama-server` |

GPT-OSS 120B possède environ 117 milliards de paramètres au total, mais seulement environ **5,1 milliards de paramètres actifs par token** grâce à son architecture MoE.

Le fichier `Q4_K_M` est réparti en deux fichiers GGUF. **Les deux fichiers sont nécessaires** au chargement du modèle.

## Télécharger le modèle

Le modèle GGUF est disponible dans le dépôt Unsloth : [unsloth/gpt-oss-120b-GGUF/Q4_K_M](https://huggingface.co/unsloth/gpt-oss-120b-GGUF/tree/main/Q4_K_M).

Pour le premier test, utiliser **Q4_K_M**. Les fichiers doivent rester dans le même répertoire :

```text
gpt-oss-120b-Q4_K_M-00001-of-00002.gguf
gpt-oss-120b-Q4_K_M-00002-of-00002.gguf
```

Le premier fichier fait environ 49,6 Go et le second environ 13,1 Go, soit environ 62,8 Go au total. Les poids ne font pas partie de ce dépôt ; respecter la licence du modèle.

## Pourquoi utiliser le CPU pour les experts

La totalité du modèle ne peut pas être placée dans les 48 Go de VRAM disponibles sur les six Polaris.

GPT-OSS étant un modèle MoE, ses poids d'experts représentent une part importante de la mémoire du modèle. `llama.cpp` permet de contrôler le placement de certains tenseurs avec `--override-tensor` (`-ot`).

Pour ce premier scénario, les tenseurs correspondant aux experts sont forcés côté CPU :

```powershell
-ot ".ffn_.*_exps.=CPU"
```

La répartition recherchée est donc :

```text
                    GPT-OSS 120B Q4_K_M
                            │
              ┌─────────────┴─────────────┐
              │                           │
       Tenseurs conservés             Tenseurs experts
          sur GPU                         sur CPU
              │                           │
       6× Polaris 8 Go                  RAM 128 Go
              │                           │
              └─────────────┬─────────────┘
                            │
                       Inférence MoE
```

Cette configuration permet de réserver autant que possible la VRAM aux calculs GPU tout en conservant les poids des experts en mémoire système.

## Premier lancement expérimental

Commencer avec un contexte relativement faible afin de vérifier séparément le chargement du modèle et la stabilité de l'inférence :

```powershell
$env:MODELE_LLAMA = "C:\chemin\vers\gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
# $env:SERVEUR_LLAMA déjà défini (voir README)

& $env:SERVEUR_LLAMA `
  -m $env:MODELE_LLAMA `
  --host 127.0.0.1 --port 8080 `
  -c 8192 `
  -ngl 99 `
  -fa on `
  --split-mode layer `
  --device Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4,Vulkan5 `
  -ot ".ffn_.*_exps.=CPU" `
  --fit off `
  -v --log-timestamps
```

Le fichier `00001-of-00002.gguf` est utilisé comme point d'entrée ; `llama.cpp` doit détecter le second fichier du modèle dans le même répertoire.

## Vérifications

Après le lancement, vérifier dans les logs :

- que les six périphériques Vulkan sont utilisés ;
- que les tenseurs `ffn_*_exps` sont bien placés sur CPU ;
- la quantité de VRAM utilisée par chaque Polaris ;
- la quantité de RAM utilisée ;
- l'absence d'erreur Vulkan ou `GGML_ASSERT` ;
- la vitesse de traitement du prompt ;
- la vitesse de génération en tokens/s.

Commencer avec `ctx = 8192`, puis augmenter progressivement si le modèle est stable :

```text
8192 → 16384 → 25600 → 32768
```

## Comparaison avec le 70B

Le scénario GPT-OSS 120B est expérimental. Il ne remplace pas la configuration actuellement validée du **Llama 3.3 70B Q4**.

La configuration 70B validée reste :

```text
6× Polaris
Llama 3.3 70B Q4_K_M
split-mode layer
ctx 25600
```

Le GPT-OSS 120B doit être évalué séparément afin de mesurer le compromis entre l'accélération Vulkan des Polaris et l'utilisation de la RAM système pour les experts.

## TurboQuant

Le support TurboQuant doit être testé séparément après validation du chargement et de l'inférence de base.

Sur Polaris, le correctif `wave64` décrit dans [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md) concerne le cache KV et n'est pas nécessaire pour vérifier le fonctionnement de base de GPT-OSS 120B.
