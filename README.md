# Revisiting Loss Modelling for Unstructured Pruning 

This is the repository containing the code of the paper "Revisiting Loss Modelling for Unstructured Pruning".


### Code structure:

 - `experiments.py` is the base code to launch experiments.
 - `log.py` contains a minimal logging utility.
 - `pruning.py` contains all the tools related to pruning.
 - `utils.py` contains the training loops and dataloaders.

### MLP on MNIST:

```
OUT_DIR='results/'

# Compute Baseline Models
for SEED in 1111 1112 1113 1114 1115; do
    python experiments.py --arch MLP --nepochs 800 --lr 0.01 --seed $SEED --pe -1 --path $OUT_DIR
done

# Pruning Experiments
for SEED in 1111 1112 1113 1114 1115; do
    MODEL0=$OUT_DIR'/arch=MLP,dec=1.0,dec_every=1,exp=False,l2=0.0005,lr=0.01,model_0=None,nepochs=800,nex=1000,pe=-1,pi=1,pm=MP,pr=0.956,reg=0.0,seed='$SEED'/best_model.pt'
    python experiments.py --arch MLP --nepochs 800 --lr 0.01 --seed $SEED --pe 0 --pr 0.98847 --pi 1 --pm MP --model_0 $MODEL0 --path $OUT_DIR
    for PM in LM QM OBD; do
        for PI in 1 14 140; do
            for LAMBDA in 0.0 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2; do
                python experiments.py --arch MLP --nepochs 800 --lr 0.01 --seed $SEED --pe 0 --pr 0.98847 --pi 1 --pm $PM --reg $LAMBDA --pi $PI --exp --model_0 $MODEL0 --path $OUT_DIR
            done
        done
    done
done
```

### VGG11 on CIFA10:

```
OUT_DIR='results/'

# Compute Baseline Models
for SEED in 1111 1112 1113 1114 1115; do
    python experiments.py --arch VGG11 --nepochs 300 --lr 0.01 --dec 0.1 --dec_every 60 --seed $SEED --pe -1 --path $OUT_DIR
done

# Pruning Experiments
for SEED in 1111 1112 1113 1114 1115; do
    MODEL0=$OUT_DIR'/arch=VGG11,dec=0.1,dec_every=60,exp=False,l2=0.0005,lr=0.01,model_0=None,nepochs=300,nex=1000,pe=-1,pi=1,pm=MP,pr=0.956,reg=0.0,seed='$SEED'/best_model.pt'
    python experiments.py --arch VGG11 --nepochs 300 --lr 0.01 --dec 0.1 --dec_every 60 --seed $SEED --pe 0 --pr 0.956 --pi 1 --pm MP --model_0 $MODEL0 --path $OUT_DIR
    for PM in LM QM OBD; do
        for PI in 1 14 140; do
            for LAMBDA in 0.0 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2; do
                python experiments.py --arch VGG11 --nepochs 300 --lr 0.01 --dec 0.1 --dec_every 60 --seed $SEED --pe 0 --pr 0.956 --pm $PM --reg $LAMBDA --pi $PI --exp --model_0 $MODEL0 --path $OUT_DIR
            done
        done
    done
done
```

### PreActResNet18 on CIFA10:

```
OUT_DIR='results/'

# Compute Baseline Models
for SEED in 1111 1112 1113 1114 1115; do
    python experiments.py --arch PreActResNet18 --nepochs 200 --lr 0.1 --dec 0.1 --dec_every 70 --seed $SEED --pe -1 --path $OUT_DIR
done

# Pruning Experiments
for SEED in 1111 1112 1113 1114 1115; do
    MODEL0=$OUT_DIR'/arch=PreActResNet18,dec=0.1,dec_every=70,exp=False,l2=0.0005,lr=0.1,model_0=None,nepochs=200,nex=1000,pe=-1,pi=1,pm=MP,pr=0.956,reg=0.0,seed='$SEED'/best_model.pt'
    python experiments.py --arch PreActResNet18 --nepochs 200 --lr 0.1 --dec 0.1 --dec_every 70 --seed $SEED --pe 0 --pr 0.956 --pi 1 --pm MP --model_0 $MODEL0 --path $OUT_DIR
    for PM in LM QM OBD; do
        for PI in 1 14 140; do
            for LAMBDA in 0.0 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2; do
                python experiments.py --arch PreActResNet18 --nepochs 200 --lr 0.1 --dec 0.1 --dec_every 70 --seed $SEED --pe 0 --pr 0.956 --pm $PM --reg $LAMBDA --pi $PI --exp --model_0 $MODEL0 --path $OUT_DIR
            done
        done
    done
done
```
