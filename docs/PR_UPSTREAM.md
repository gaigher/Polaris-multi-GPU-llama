# Contribution upstream

Issue : [#330](https://github.com/TheTom/llama-cpp-turboquant/issues/330) — branche cible `feature/turboquant-kv-cache`.

Le fichier fourni pour le build local est `patches/polaris-subgroup.patch` (deux hunks).

**PR recommandée :** n’envoyer que le hunk `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (taille de subgroup conditionnelle).

Le hunk CMake (`ggml/src/ggml-vulkan/CMakeLists.txt`, compilateur hôte pour `vulkan-shaders-gen`) est un correctif Windows local — commit / PR séparée si demandé.

**Titre suggéré :**  
`fix(vulkan): conditional subgroup size for TurboQuant SET_ROWS on wave64-only GPUs`

Contexte technique et mesures : [TurboQuant_POLARIS.md](TurboQuant_POLARIS.md).
