steps=500
lr=0.01
for seed in {1..5}; do
    for width in 100 200 300 400; do
        for depth in 1 2 3; do
            dir_name=ablation-$width-$depth
            poetry run python3 tests_NEXT/heat2d_NEXT.py \
            --dir $dir_name \
            --depth $depth \
            --width $width \
            --seed $seed \
            --lr $lr \
            --steps $steps
        done
    done
done