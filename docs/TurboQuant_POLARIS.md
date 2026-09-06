# TurboQuant sur AMD Polaris (Vulkan) — optionnel

> Optimisation **optionnelle** du cache KV (`turbo3`) pour le setup [6× Polaris / Llama 70B](../README.md). Le 70B multi-GPU fonctionne aussi sans TurboQuant.

Correctifs et scripts pour exécuter le **cache KV TurboQuant** sur **AMD GCN Polaris** (RX 570 / RX 580, wave64) avec [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).

Les binaires Vulkan Windows précompilés fixent souvent `requiredSubgroupSize = 32` (RDNA4). Polaris ne supporte que le **wave64** → `GGML_ASSERT` au chargement. Ce dépôt fournit le correctif.

Suivi upstream : [issue #330](https://github.com/TheTom/llama-cpp-turboquant/issues/330).

## Build (patch + compile)



### 1. Prérequis

- Visual Studio 2022 (Développement Desktop C++)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
- CMake, Git, Ninja (recommandé)



### 2. Cloner amont + patch

Depuis la **racine de ce dépôt** (déjà cloné) :

```powershell
$CheminTheTom = Join-Path (Split-Path $PWD -Parent) "llama-cpp-turboquant"

git clone --branch feature/turboquant-kv-cache --depth 1 `
  https://github.com/TheTom/llama-cpp-turboquant.git `
  $CheminTheTom

.\scripts\appliquer_patch.ps1 -CheminDepot $CheminTheTom
```

`appliquer_patch.ps1` applique `patches/polaris-subgroup.patch`. Sans `-CheminDepot`, le chemin par défaut est `%USERPROFILE%\Documents\llama-cpp-turboquant`.

### 3. Compiler

Toujours depuis **ce** dépôt, en Developer PowerShell for VS 2022 :

```powershell
.\scripts\compiler_windows.ps1 -CheminDepot $CheminTheTom
$env:SERVEUR_LLAMA = Join-Path $CheminTheTom "build\bin\llama-server.exe"
```

Ensuite, lancer le 70B avec TurboQuant : [README](../README.md) (section *Optimisation optionnelle*).

## Problème (wave64)

Les chemins d’écriture KV TurboQuant utilisent `requiredSubgroupSize = 32` ([PR TheTom #160](https://github.com/TheTom/llama-cpp-turboquant/pull/160)) pour le packing ballot wave32 RDNA4 dans les shaders SET_ROWS.

**AMD GCN Polaris** (RX 570, RX 580) n’expose que des subgroups **wave64** :

```
subgroup_min_size = subgroup_max_size = 64
```

Avec un pin inconditionnel à 32, llama.cpp assert à la création du pipeline :

```
GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size
         && required_subgroup_size <= device->subgroup_max_size) failed
```

Les zip Windows précompilés incluant la PR #160 **plantent donc sur turbo2/turbo3/turbo4** alors que `f16` / `q4_0` fonctionnent encore.

## Correctif

Utiliser une taille de subgroup conditionnelle avant la macro SET_ROWS :

```cpp
const uint32_t turbo_set_rows_subgroup =
    (device->subgroup_size_control &&
     32u >= device->subgroup_min_size && 32u <= device->subgroup_max_size)
        ? 32u : 0u;
```

Remplacer le `32u` codé en dur sur les lignes SET_ROWS turbo2/3/4/tq4_1s par `turbo_set_rows_subgroup`.

Sur Polaris → `0` (pas de pin), chemin shader wave64 de la [PR #243](https://github.com/TheTom/llama-cpp-turboquant/pull/243).

Sur RDNA4 avec subgroups flexibles → `32` si supporté.

## Rôle du patch

`patches/polaris-subgroup.patch` (guide / build local) combine **deux** changements :

1. `ggml-vulkan.cpp` — taille de subgroup conditionnelle pour les pipelines turbo SET_ROWS :
  - utiliser `32` seulement si le GPU la supporte dans `[subgroup_min_size, subgroup_max_size]`
  - sinon `0` (pas de pin) → wave64 natif sur Polaris
2. `CMakeLists.txt` — transmettre le compilateur C/C++ hôte à l’ExternalProject `vulkan-shaders-gen` (corrige certains builds VS/Ninja sous Windows)

Pour une PR upstream, extraire les hunks **séparément** depuis `patches/polaris-subgroup.patch` :

- `ggml-vulkan.cpp` — subgroup seul (recommandé pour la PR)
- `CMakeLists.txt` Vulkan — CMake Windows (optionnel, commit séparé)

Détail : [PR_UPSTREAM.md](PR_UPSTREAM.md).

## Note CMake (Windows)

Certaines compilations échouent sur `vulkan-shaders-gen` car le sous-build ExternalProject n’hérite pas du compilateur hôte. Passer `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` depuis le projet parent corrige le flux Ninja + `vcvars64` testé ici.

## Llama 70B / GQA

Sur Llama 70B (GQA 8:1), TheTom active l’**auto-asymétrique** : K passe en `q8_0`, V reste en `turbo3`.

Avec `--cache-type-k turbo3 --cache-type-v turbo3`, TheTom journalise typiquement :

```
auto-asymmetric: GQA ratio 8:1 — upgrading K from turbo3 to q8_0
```

C’est voulu (qualité). Config recommandée : `--cache-type-k q8_0 --cache-type-v turbo3` (explicite, même résultat).  
Désactiver l’auto-asymétrique : `TURBO_AUTO_ASYMMETRIC=0` (déconseillé pour le 70B).

Mesures et lancement multi-GPU : [README](../README.md).

## Contribution upstream

Ouvrez une PR sur **TheTom/llama-cpp-turboquant** avec le hunk subgroup de `patches/polaris-subgroup.patch` si possible. Voir [PR_UPSTREAM.md](PR_UPSTREAM.md) et l’issue [#330](https://github.com/TheTom/llama-cpp-turboquant/issues/330).

## Références

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) branche `feature/turboquant-kv-cache`
- [Discussion llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- PR [#160](https://github.com/TheTom/llama-cpp-turboquant/pull/160), [#243](https://github.com/TheTom/llama-cpp-turboquant/pull/243)

