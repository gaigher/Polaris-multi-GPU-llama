# Runbook PR — Vulkan moe-cache (faux UMA / Polaris)

Cible : [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), branche de base `feature/turboquant-kv-cache`.

Fichier concerné : `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp`  
Repo local : `%USERPROFILE%\Documents\llama-cpp-turboquant`

**PR ouverte :** [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) — `gaigher:patch-1` → `feature/turboquant-kv-cache` (commit `a73fd8a`, correctness only)

PR voisine (autre sujet) : [PR_UPSTREAM.md](PR_UPSTREAM.md) — subgroup SET_ROWS Polaris.

---

## État (checklist)

### Fait

- [x] Diagnostiquer `fail_alloc` ~508 MiB en `host-mapped` malgré VRAM libre (Polaris, aperture ~256 MiB)
- [x] Patch : n’activer host-mapped que si le tas UMA ≥ budget ; sinon **device-local + staging**
- [x] Override env : `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1`
- [x] Rebuild `ggml-vulkan.dll` (Ninja Release, `GGML_VULKAN=ON`)
- [x] Probe S-23 `--moe-cache 512` : `device-local + staging`, `session ready`, `pools=1`, `fail_alloc=0`, hits ~49 %
- [x] Bench S-23 `@8192` **moe-cache=off** (3 runs) — baseline locale
- [x] Bench S-23 `@8192` **moe-cache=512** (3 runs) — **régression forte** (voir tableau)
- [x] Mettre à jour [GPT-OSS_120B.md](GPT-OSS_120B.md) avec chiffres off vs 512 + logs clés
- [x] ~~A/B~~ `@32768` — inutile tant que 512 MiB thrash
- [x] PR GitHub **correctness only** : [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401)  
  (via éditeur web GitHub + fork `gaigher`, branche `patch-1` — pas via push local `fix/vulkan-moe-cache-discrete-uma`)

### Reste à faire

- [x] Stash WIP `wip-other-polaris` restauré → `ggml-vulkan.cpp` + `CMakeLists.txt` en working tree (branche locale `feature/turboquant-kv-cache`)
- [ ] **Review / merge** de [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) — **côté mainteneur** (`TheTom`) ; mergeable=true, pas de droits push/merge pour `gaigher`
- [x] Prod Polaris 8 Go : **`--moe-cache off`** documenté / commandes de lancement (voir [GPT-OSS_120B.md](GPT-OSS_120B.md) + [Décision tok/s](#décision-toks-après-bench))

**État git (référence)** :

| Lieu | État |
| ---- | ---- |
| GitHub PR | [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) — head `gaigher:patch-1`, commit `a73fd8a`, **ouverte** |
| Local `%USERPROFILE%\Documents\llama-cpp-turboquant` | `feature/turboquant-kv-cache` ; WIP subgroup/CMake **restauré** ; patch moe-cache local retiré (doublon de la PR) ; `dist.tar.gz*` toujours untracked |
| Stash | `wip-other-polaris` **poppé** (vide) |



### Hors scope de cette PR (ne pas mélanger)


| Fichier local modifié                    | Action                                             |
| ---------------------------------------- | -------------------------------------------------- |
| `ggml-vulkan.cpp` (subgroup SET_ROWS)    | PR séparée — voir [PR_UPSTREAM.md](PR_UPSTREAM.md) |
| `CMakeLists.txt` (compiler hôte shaders) | Correctif Windows local — PR séparée si besoin     |
| `tools/ui/dist.tar.gz*`                  | **Ne pas committer**                               |


---



## Contenu de la PR (un seul fichier)



### Titre suggéré

`fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache on discrete AMD`

### Corps suggéré

```markdown
## Summary
- Discrete AMD (Polaris, etc.) often exposes a small DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT heap (~256 MiB).
- Vulkan moe-cache treated that as UMA and allocated the expert slab there → `failed to allocate ~508 MiB` even with free VRAM.
- Only enable host-mapped when that heap size ≥ configured budget; otherwise use device-local + staging.
- Override: `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1`.

## Evidence (6× RX 580 8GB, GPT-OSS 120B Q4_K_M, profile S-23)
**Before:** host-mapped → fail_alloc, pools=0  
**After:** `UMA heap=256 MiB < need=512 MiB; using device-local + staging` → pool 508 MiB, fail_alloc=0, hits≈49%

## Performance note (correctness-only PR)
Local A/B `@8192` ×3 on Polaris 8GB: `--moe-cache off` gen ~3352 tok/s vs `--moe-cache 512` gen ~369 tok/s (~9× slower; pool thrashing). This PR fixes allocation correctness; it does **not** claim a production perf win on discrete Polaris — keep `--moe-cache off` until a better cache strategy (VRAM headroom, budget, HOT_USES).

## Test plan
- [ ] Rebuild with `-DGGML_VULKAN=ON`
- [ ] `--moe-cache 512` on discrete AMD: log shows device-local + staging, `session ready`, `pool[...]`, no fail_alloc
- [ ] True UMA / large aperture: still host-mapped when heap ≥ budget
- [ ] `GGML_VK_MOE_CACHE_FORCE_DEVICE_LOCAL=1` forces device-local
- [x] tok/s A/B: `--moe-cache off` vs `512` on same `-ot` (regression documented above)
```

---



## Étapes manuelles (sans invite de commande)

Procédure via **Cursor** (ou VS Code) + **GitHub dans le navigateur**. Aucun PowerShell requis.

### Liens utiles

| Étape | Lien |
| ----- | ---- |
| **PR ouverte** | [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) |
| Commit PR | [`a73fd8a`](https://github.com/TheTom/llama-cpp-turboquant/commit/a73fd8a0193fd907015ab7b6a52fe727fbbfb001) |
| Head (fork) | [`gaigher:patch-1`](https://github.com/gaigher/llama-cpp-turboquant/tree/patch-1) |
| Dépôt upstream | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |
| Branche base | [feature/turboquant-kv-cache](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache) |
| Compare (historique / re-PR) | [base ← gaigher:patch-1](https://github.com/TheTom/llama-cpp-turboquant/compare/feature/turboquant-kv-cache...gaigher:patch-1?expand=1) |
| Fichier modifié (référence upstream) | [ggml-vulkan-moe-cache.cpp](https://github.com/TheTom/llama-cpp-turboquant/blob/feature/turboquant-kv-cache/ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp) |
| Toutes les PR ouvertes | [Pull requests](https://github.com/TheTom/llama-cpp-turboquant/pulls) |

**Dossier local à ouvrir dans Cursor :**  
`%USERPROFILE%\Documents\llama-cpp-turboquant`

---

### 1. Ouvrir le bon repo et la bonne branche

1. **Fichier → Ouvrir le dossier…** → `Documents\llama-cpp-turboquant`
2. Barre d’état en bas à gauche : la branche doit être **`fix/vulkan-moe-cache-discrete-uma`**
3. Si ce n’est pas la bonne branche : clic sur le nom de branche → choisir `fix/vulkan-moe-cache-discrete-uma` (ou **Create new branch…** depuis `feature/turboquant-kv-cache` si elle n’existe pas encore)

---

### 2. Vérifier qu’un seul fichier part dans la PR

1. **Ctrl+Shift+G** → panneau **Contrôle de code source**
2. Section **Indexées** (*Staged Changes*) : **uniquement**  
   `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp`
3. **Ne pas inclure** dans ce commit :
   - `ggml-vulkan.cpp` (subgroup — autre PR)
   - `CMakeLists.txt` (correctif Windows local)
   - `tools/ui/dist.tar.gz*` (**jamais**)
4. Si d’autres fichiers apparaissent dans *Modifications* :
   - clic droit → **Conserver les modifications** (*Stash*) sur les fichiers hors scope, **ou**
   - ne pas les cocher / les retirer des indexées (**Retirer de l’index**)

---

### 3. Commit (Cursor)

1. Dans **Indexées**, vérifier le diff de `ggml-vulkan-moe-cache.cpp`
2. Champ **Message** (titre, 1re ligne) :

   `fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache`

3. Champ **Description** (corps du commit, optionnel mais recommandé) :

   ```
   Discrete AMD exposes a tiny DEVICE_LOCAL|HOST_VISIBLE heap that was
   misdetected as UMA, so expert slab alloc failed despite free VRAM.
   Prefer device-local + staging when UMA heap < budget; allow FORCE_DEVICE_LOCAL.
   ```

4. Bouton **Valider** (*Commit*) — **Commit** (pas « Commit & Push » tant que tu n’as pas relu)

---

### 4. Push la branche (Cursor)

1. Après le commit : bouton **Synchroniser les modifications** (*Sync*) ou **Publier la branche** (*Publish Branch*) en bas à gauche
2. Si Cursor demande le remote : **`origin`** → `https://github.com/TheTom/llama-cpp-turboquant.git`  
   (ou ton fork si tu n’as pas les droits push sur `TheTom` — dans ce cas, adapte le lien compare à ton fork)
3. Attendre la fin du push (barre de progression)

**Alternative — GitHub Desktop :**  
Ouvrir le même dossier → onglet **Changes** → commit identique → **Push origin**.

---

### 5. Créer la PR sur GitHub (navigateur)

1. Ouvrir le lien compare (tableau ci-dessus) :  
   [feature/turboquant-kv-cache ← fix/vulkan-moe-cache-discrete-uma](https://github.com/TheTom/llama-cpp-turboquant/compare/feature/turboquant-kv-cache...fix/vulkan-moe-cache-discrete-uma?expand=1)
2. Vérifier :
   - **base** = `feature/turboquant-kv-cache`
   - **compare** = `fix/vulkan-moe-cache-discrete-uma`
   - **1 file changed** : `ggml-vulkan-moe-cache.cpp` seulement
3. **Create pull request**
4. **Title** (copier-coller) :

   `fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache on discrete AMD`

5. **Description** : copier le bloc *Corps suggéré* de la section [Contenu de la PR](#contenu-de-la-pr-un-seul-fichier) (résumé + evidence + note perf + test plan)
6. **Create pull request** (définitif)

Si GitHub affiche « There isn’t anything to compare » : la branche n’est pas encore poussée → reprendre l’étape 4.

---

### 6. Après la PR — restaurer le WIP local (subgroup / CMake)

Le stash `wip-other-polaris` contient les autres modifications. Sans terminal :

1. **Ctrl+Shift+P** → taper **`Git: Pop Stash`** → Entrée
2. Choisir le stash **`wip-other-polaris`** (ou le plus récent nommé ainsi)
3. Vérifier dans le contrôle de code source que `ggml-vulkan.cpp` / `CMakeLists.txt` sont revenus en *Modifications* **non indexées** (pas dans la PR moe-cache)

**GitHub Desktop :** menu **Branch** → **Stashed changes** → **Restore**.

---

### 7. Cocher la checklist ici

- [x] Commit créé — `a73fd8a` sur `gaigher:patch-1` (éditeur web ; équivalent local = branche `fix/vulkan-moe-cache-discrete-uma`)
- [x] Branche disponible sur le fork (`patch-1`)
- [x] PR ouverte : [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) vers `feature/turboquant-kv-cache` (correctness only)
- [x] Stash WIP restauré
- [x] En prod Polaris 8 Go : **`--moe-cache off`** (voir [Décision tok/s](#décision-toks-après-bench) / [GPT-OSS_120B.md](GPT-OSS_120B.md))
- [ ] Merge [#401](https://github.com/TheTom/llama-cpp-turboquant/pull/401) par le mainteneur

---



## Étapes Git (terminal — optionnel)

> Équivalent CLI de [§ Étapes manuelles](#étapes-manuelles-sans-invite-de-commande). Ne committer / pusher que si tu le demandes explicitement.

1. **Partir de la branche cible à jour**
  ```powershell
   cd $env:USERPROFILE\Documents\llama-cpp-turboquant
   git fetch origin
   git checkout feature/turboquant-kv-cache
   git pull --ff-only origin feature/turboquant-kv-cache
  ```
2. **Branche dédiée**
  ```powershell
   git checkout -b fix/vulkan-moe-cache-discrete-uma
  ```
3. **Stasher / isoler les autres diffs**
  ```powershell
   git stash push -m "wip-other" -- ggml/src/ggml-vulkan/CMakeLists.txt ggml/src/ggml-vulkan/ggml-vulkan.cpp
   git add ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp
   git status   # doit montrer UNIQUEMENT moe-cache.cpp staged
  ```
4. **Commit**
  ```powershell
   git commit -m @"
   fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache

   Discrete AMD exposes a tiny DEVICE_LOCAL|HOST_VISIBLE heap that was
   misdetected as UMA, so expert slab alloc failed despite free VRAM.
   Prefer device-local + staging when UMA heap < budget; allow FORCE_DEVICE_LOCAL.
   "@
  ```
5. **Push + PR** (nécessite `gh` installé et auth)
  ```powershell
   git push -u origin HEAD
   gh pr create --base feature/turboquant-kv-cache --title "fix(vulkan): reject small HOST_VISIBLE aperture for moe-cache on discrete AMD" --body-file path\to\corps.md
  ```
   Sans `gh` : pousser la branche puis ouvrir la PR dans l’UI GitHub vers `feature/turboquant-kv-cache`.
6. **Restaurer le WIP local**
  ```powershell
   git stash pop
  ```

---



## Preuves locales à joindre / citer


| Artefact         | Chemin typique                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Probe alloc OK   | `logs/moe-alloc-S-b512-*.log` — lignes `device-local + staging`, `pool[0]`, `fail_alloc` absent |
| CSV probe        | `bench_results/moe-alloc-S-*.csv`                                                               |
| Bench off vs 512 | `bench_results/abc-*.csv` + résumé `abc-*.txt`                                                  |
| Script bench     | `scripts/bench_moe_cache_abc.ps1` (`-ProfileFilter S -MoeCache off,512 -Runs 3`)                |
| Script probe     | `scripts/probe_moe_cache_alloc.ps1` (`-Profile S -Budgets 512`)                                 |


Lignes log de référence (après patch) :

```
[moe-cache] Vulkan: UMA heap=256 MiB < need=512 MiB; using device-local + staging (discrete/aperture path)
[moe-cache] Vulkan session ready (budget=512 MiB, device-local)
[moe-cache] Vulkan0 pool[0]: type=mxfp4 ... total=508 MiB
[moe-cache] enabled: first pool allocated on Vulkan0
```

---



## Décision tok/s (après bench)


| Résultat                          | Action PR                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `512` ≥ `off` (surtout gen tok/s) | Mentionner le gain dans la PR                                                                                        |
| `512` ≈ `off`                     | PR = **correctness** (cache utilisable) ; pas de claim perf                                                          |
| `512` ≪ `off`                     | PR correctness quand même ; noter thrashing / coût staging ; pistes : budget plus grand, HOT_USES, moins d’évictions |




### Résultat bench final `@8192` ×3 — `abc-20261004-172308`


| MoeArg  | PromptAvg | GenAvg    | GenMin–Max  | Cache               |
| ------- | --------- | --------- | ----------- | ------------------- |
| **off** | **2,725** | **3,352** | 2,998–3,530 | n/a                 |
| **512** | **0,353** | **0,369** | 0,363–0,374 | hits 49,8 % Vulkan0 |


→ gen **~9× plus lent** avec cache (pool plein, thrashing fills/évictions).  
**PR = correctness only** (alloc device-local OK). En prod Polaris 8 Go : garder `--moe-cache off` jusqu’à meilleure stratégie (marge V0, budget, HOT_USES).
