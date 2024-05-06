#!/bin/bash

# python -u pvae/main.py \
#     --model mnist \
#     --manifold Euclidean \
#     --save-freq 0 \
#     --epochs 80 \
#     --batch-size 128 \
#     --lr 5e-4 \
#     --latent-dim 2 \
#     --posterior Normal \
#     --hidden-dim 600 \
#     --enc Wrapped \
#     --dec Wrapped \
#     --prior Normal \
#     --iwae-samples 5000

# models...
# like N-VAE, dim = 2... iwae mlik in paper = 144.5 +/- 0.4
# tuesday
# /workspaces/pvae/experiments/2024-04-02T22_55_40.704953zt_bl41g/run.log 
# - iwae mlik after 80 epochs = 142.8959
# wednesday
# /workspaces/pvae/experiments/2024-04-03T16_29_05.924731iitut_rj 
# - iwae (k=10) after 58 epochs = 147.1126

python -u pvae/main.py \
    --model mnist \
    --manifold PoincareBall \
    --save-freq 0 \
    --epochs 80 \
    --batch-size 128 \
    --lr 5e-4 \
    --latent-dim 2 \
    --c 1.4 \
    --posterior WrappedNormal \
    --hidden-dim 600 \
    --enc Wrapped \
    --dec Geo \
    --prior WrappedNormal \
    --iwae-samples 5000

# models...
# paper
# - iwae (k=5000) = 144.0±0.6
# wednesday
# /workspaces/pvae/experiments/2024-04-03T16_58_22.1338573183fuxu/run.log 
# - iwae (k=10) after 41 epochs = 147.1028
# /workspaces/pvae/experiments/2024-04-03T18_11_02.088532yv975u_8/run.log
# - iwae (k=10) after 5 epochs = 152.8002
# - iwae (k=10) after 26 epochs = 147.9390
# - iwae (k=10) after 47 epochs = 146.4840
