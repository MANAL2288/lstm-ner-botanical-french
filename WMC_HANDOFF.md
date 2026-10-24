# WMC — livraison du duo Said & MS

## Périmètre
Cellule WMC des équations 7–12 du sujet, intégrée au pipeline commun. La comparaison des trois architectures, le rapport et les slides restent au groupe.

## Implémentation
`WMCCell` utilise les projections d'entrée et d'état caché de Vanilla, plus trois matrices de mémoire H×H sans biais supplémentaire. Les portes i et f utilisent tanh(Wc c précédent). La nouvelle mémoire est calculée avant o, qui utilise tanh(Wco c nouveau). Les états et dimensions restent compatibles avec le déroulement bidirectionnel et le masquage existants. Aucun nn.LSTM/nn.LSTMCell n'est utilisé dans la cellule.

## Protocole inchangé
Tâche fine, mêmes splits et tokenisation, embeddings 100 + casse 8, H=128, une couche bidirectionnelle, dropout 0.3, Adam, lr=0.001, batch 32, initialisation uniforme commune, gradient clipping 5. Seeds 1/2/3, au plus 60 epochs, patience 10 sur F1 validation, meilleur état validation évalué sur test. Le lr=0.01 de la meilleure expérience Vanilla n'est pas substitué à la baseline commune.

## Paramètres
Une cellule : 4(DH + H² + H) + 3H². Pour D=108, H=128 : 170 496. Deux directions : 340 992 paramètres récurrents, soit 98 304 de plus que Vanilla. Avec le vocabulaire commun et les mêmes sorties, total attendu : 898 339 (à confirmer dans metrics.json après chargement des données).

## Exécution
Option simple : ouvrir notebooks/FloraNER_WMC_colab.ipynb dans Colab, sélectionner T4 GPU et Tout exécuter. Le notebook embarque le code et les tests, sauvegarde dans MyDrive/floraner_wmc_results_e60, entraîne uniquement WMC et télécharge un zip des résultats. Accepter le montage de son Drive dans Colab. Les expériences finies sont sautées, mais une expérience interrompue recommence depuis sa seed.

En local depuis la racine :
```
pip install -r requirements.txt
python -m pytest tests -q
python src/download_data.py
python src/sweep.py --cell wmc --study baseline --seeds 1 2 3 --epochs 60 --patience 10
python src/summarize_wmc.py
```

## Fichiers à transmettre après entraînement
- results/runs/fine_wmc_baseline_s1, s2, s3 : config.json, history.json, metrics.json.
- Les lignes WMC de results/summary.jsonl.
- wmc_summary.md : résultats individuels, moyenne et écart-type échantillonnal, paramètres.
- Le code et les tests via une Pull Request de wmc-lstm vers main, sans push direct sur main.

Les résultats numériques d'entraînement ne sont pas inventés : consulter le statut de validation joint pour savoir ce qui a réellement été exécuté.
