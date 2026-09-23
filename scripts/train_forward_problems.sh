
#!/bin/bash

directory="paper-experiment"

echo "Training the 1D Heat equation"
for seed in {1..10}; do
    echo NEXT seed $seed
    poetry run python3 tests_NEXT/heat1d_NEXT.py \
    --dir $directory \
    --seed $seed \
    --steps 500 \
    --solver etd4 \
    --lr 0.02

    for model in PINN QRes FLS; do
        echo $model seed $seed
        poetry run python3 tests_NEXT/heat1d_mlp_models.py \
        --dir $directory \
        --seed $seed \
        --model $model \
        --steps 5000 \
        --lr 0.001
    done
done

echo "Training the Burgers equation"
for seed in {1..10}; do
    echo NEXT seed $seed
    poetry run python3 \
    tests_NEXT/burgers1d_NEXT.py \
    --dir $directory \
    --seed $seed \
    --steps 500 \
    --lr 0.02 \
    --solver etd4

    for model in PINN QRes FLS; do
        echo $model seed $seed
        poetry run python3 \
        tests_NEXT/burgers1d_mlp_models.py \
        --dir $directory \
        --seed $seed \
        --model $model \
        --steps 5000 \
        --lr 0.001
    done
done

echo "Training the wave equation"
for seed in {1..10}; do
    echo NEXT seed $seed
    poetry run python3 \
    tests_NEXT/wave1d_NEXT.py \
    --dir $directory \
    --seed $seed \
    --steps 1000 \
    --lr 0.02 \
    --solver etd4 \
    --timesteps 100

    for model in PINN QRes FLS; do
        echo $model seed $seed
        poetry run python3 \
        tests_NEXT/wave1d_mlp_models.py \
        --dir $directory \
        --seed $seed \
        --model $model \
        --steps 5000 \
        --lr 0.001
    done
done

echo "Training the Korteweg-de Vries equation"
for seed in {1..10}; do
    for solver in etd1 etd4 KT; do
        echo NEXT seed $seed $solver
        poetry run python3 \
        tests_NEXT/kdv1d_ZK_NEXT.py \
        --dir $directory \
        --seed $seed \
        --steps 1000 \
        --lr 0.01 \
        --solver $solver
    done
    for model in PINN QRes FLS; do
        echo $model seed $seed
        poetry run python3 \
        tests_NEXT/kdv1d_ZK_mlp_models.py \
        --dir $directory \
        --seed $seed \
        --model $model \
        --steps 10000 \
        --lr 0.001
    done
done

echo "Training the 2D Heat equation"
for seed in {1..10}; do
    for solver in etd1 etd4; do
        echo NEXT seed $seed $solver
        poetry run python3 \
        tests_NEXT/heat2d_NEXT.py \
        --dir $directory \
        --seed $seed \
        --steps 500 \
        --depth 2 \
        --width 100 \
        --lr 0.01 \
        --solver $solver
    done

    for model in PINN QRes FLS; do
        echo $model seed $seed
        poetry run python3 \
        tests_NEXT/heat2d_mlp_models.py \
        --dir $directory \
        --seed $seed \
        --model $model \
        --steps 20000 \
        --lr 0.001
    done
done
        