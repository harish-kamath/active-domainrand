#! /bin/bash
# To check results, run "tmux a -t ADR". Kill the session before you start a new one.

tmux new-session -d -s ADR_lunar

SEEDS=(100 200 300)
ENV_TYPE="lunar"
REVAL_ENV_ID="LunarLanderRandomized-v0"

for SEED in "${SEEDS[@]}"
do
    tmux new-window -d -n baseline_$SEED
    tmux new-window -d -n udr_$SEED
    tmux new-window -d -n adrold_$SEED
    tmux new-window -d -n adrnew_$SEED
    tmux new-window -d -n adrpd_$SEED

    tmux send -t ADR_lunar:baseline_$SEED.0 'conda activate curiosity' Enter
    tmux send -t ADR_lunar:udr_$SEED.0 'conda activate curiosity' Enter
    tmux send -t ADR_lunar:adrold_$SEED.0 'conda activate curiosity' Enter
    tmux send -t ADR_lunar:adrnew_$SEED.0 'conda activate curiosity' Enter
    tmux send -t ADR_lunar:adrpd_$SEED.0 'conda activate curiosity' Enter

    # SLURM
    # srun --gres=gpu:1 -p short
    # Exclude List is (from BAD to OKAY): calculon,bb8,walle
    
    SLURM_PREFIX="srun --exclude=calculon,bb8,walle --gres=gpu:1 -c 6  -p short python -m experiments.domainrand.experiment_driver"

    # Baseline
    tmux send -t ADR_lunar:baseline_$SEED.0 "$SLURM_PREFIX $ENV_TYPE\
        --initial-svpg-steps=1e6 --continuous-svpg --experiment-name=unfreeze-policy --freeze-svpg --freeze-discriminator --seed=$SEED\
        --agent-name=baseline --experiment-prefix=baseline" Enter

    # UDR
    tmux send -t ADR_lunar:udr_$SEED.0 "$SLURM_PREFIX $ENV_TYPE\
        --initial-svpg-steps=1e6 --continuous-svpg --experiment-name=unfreeze-policy --freeze-svpg --freeze-discriminator\
        --agent-name=udr --experiment-prefix=udr --seed=$SEED\
        --randomized-env-id=$REVAL_ENV_ID" Enter

    # ADR Old
    tmux send -t ADR_lunar:adrold_$SEED.0 "$SLURM_PREFIX $ENV_TYPE\
        --continuous-svpg --experiment-name=unfreeze-policy\
        --agent-name=adrold --experiment-prefix=adrold --seed=$SEED\
        --randomized-env-id=$REVAL_ENV_ID" Enter
    
    # ADR New
    tmux send -t ADR_lunar:adrnew_$SEED.0 "$SLURM_PREFIX $ENV_TYPE\
        --continuous-svpg --experiment-name=unfreeze-policy\
        --agent-name=adrnew --experiment-prefix=adrnew --seed=$SEED\
        --randomized-env-id=$REVAL_ENV_ID\
        --use-new-discriminator modeladv" Enter
    
    # ADR PD
    tmux send -t ADR_lunar:adrpd_$SEED.0 "$SLURM_PREFIX $ENV_TYPE\
        --continuous-svpg --experiment-name=unfreeze-policy\
        --agent-name=adrpd --experiment-prefix=adrpd --seed=$SEED\
        --randomized-env-id=$REVAL_ENV_ID\
        --use-new-discriminator perfdiff" Enter
done