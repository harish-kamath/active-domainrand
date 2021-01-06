#!/bin/sh
# To check results, run "tmux a -t ADR". Kill the session before you start a new one.

tmux new-session -d -s ADR_pusher
tmux new-window -d -n baseline
tmux new-window -d -n udr
tmux new-window -d -n adrold
tmux new-window -d -n adrnew

tmux send -t ADR_pusher:baseline.0 'conda activate curiosity' Enter
tmux send -t ADR_pusher:udr.0 'conda activate curiosity' Enter
tmux send -t ADR_pusher:adrold.0 'conda activate curiosity' Enter
tmux send -t ADR_pusher:adrnew.0 'conda activate curiosity' Enter

# SLURM
# srun --gres=gpu:1 -p short

ENV_TYPE="pusher"
REVAL_ENV_ID="Pusher3DOFRandomized-v0"

# Baseline
tmux send -t ADR_pusher:baseline.0 "srun --gres=gpu:1 -p short python -m experiments.domainrand.experiment_driver $ENV_TYPE\
    --initial-svpg-steps=1e6 --continuous-svpg --experiment-name=unfreeze-policy --freeze-svpg --freeze-discriminator\
    --agent-name=baseline --experiment-prefix=baseline" Enter

# UDR
tmux send -t ADR_pusher:udr.0 "srun --gres=gpu:1 -p short python -m experiments.domainrand.experiment_driver $ENV_TYPE\
    --initial-svpg-steps=1e6 --continuous-svpg --experiment-name=unfreeze-policy --freeze-svpg --freeze-discriminator\
    --agent-name=udr --experiment-prefix=udr\
    --randomized-env-id=$REVAL_ENV_ID" Enter

# ADR Old
tmux send -t ADR_pusher:adrold.0 "srun --gres=gpu:1 -p short python -m experiments.domainrand.experiment_driver $ENV_TYPE\
    --continuous-svpg --experiment-name=unfreeze-policy\
    --agent-name=adrold --experiment-prefix=adrold\
    --randomized-env-id=$REVAL_ENV_ID" Enter
# ADR New
tmux send -t ADR_pusher:adrnew.0 "srun --gres=gpu:1 -p short python -m experiments.domainrand.experiment_driver $ENV_TYPE\
    --continuous-svpg --experiment-name=unfreeze-policy\
    --agent-name=adrnew --experiment-prefix=adrnew\
    --randomized-env-id=$REVAL_ENV_ID\
    --use-new-discriminator" Enter