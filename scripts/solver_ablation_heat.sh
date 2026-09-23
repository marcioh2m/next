directory=solver-ablation
for solver in etd1 etd4 rk4; do
    for seed in {1..5}; do
        echo $solver $seed
        poetry run python3 tests_NEXT/heat2d_NEXT.py \
        --dir $directory \
        --seed $seed \
        --solver $solver \
        --lr 0.01 \
        --steps 500 \
        --depth 2 \
        --width 100
    done
done


