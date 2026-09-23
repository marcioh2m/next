## Solves the Burgers inverse problem for random values of the starting parameters \lambda

#!/bin/bash

PARAM_VALUES="data/burgers1d_inverse/param_values.txt"

### Uncomment this to generate a batch of starter values of lambda_1 and lambda_2
# python3 -c "
# import random
# random.seed(314159)
# for _ in range(10):
#     lambda_1 = random.uniform(0.1, 2.0)
#     lambda_2 = random.uniform(0.01, 0.20)
#     print(f'{lambda_1} {lambda_2}')
# " > "$PARAM_VALUES"

echo "Initial parameter values randomly generated and saved to $PARAM_VALUES."

true_nu=0.1

samples=($(tr -d '\r' < "$PARAM_VALUES"))
for data_points in 0; do
    for noise_level in 0 0.01 0.05 0.10; do
        seed=1
        for ((i=0; i<${#samples[@]}; i+=2)); do
            lambda_1="${samples[i]}"
            lambda_2="${samples[i+1]}"
            echo "lambda_1 = $lambda_1"
            echo "lambda_2 = $lambda_2"
            echo "seed = $seed"
            echo "$data_points points"
            echo "noise = $noise_level"
            poetry run python3 \
            tests_NEXT_inverse/inverse_burgers1d_NEXT.py \
            --dir burgers_inverse-endpoint-$noise_level-noise \
            --seed $seed \
            --device cuda:0 \
            --physics_weight 1 \
            --data_weight 10 \
            --data_points $data_points \
            --lambda_0 $lambda_1 \
            --nu_0 $lambda_2 \
            --nu $true_nu \
            --trainable_L \
            --steps 500 \
            --noise_level $noise_level \
            --grid_factor 4 \
            --data_regime endpoint
            

            ((seed++))
            
        done < "$PARAM_VALUES"
    done
done
for data_points in 0; do
    for noise_level in 0 0.01 0.05 0.10; do
        for model in QRes PINN FLS; do
            seed=1
            for ((i=0; i<${#samples[@]}; i+=2)); do
                lambda_1="${samples[i]}"
                lambda_2="${samples[i+1]}"
                echo "lambda_1 = $lambda_1"
                echo "lambda_2 = $lambda_2"
                echo "seed = $seed"
                echo "model = $model"
                echo "$data_points points"
                echo "noise = $noise_level"
                poetry run python3 \
                tests_NEXT_inverse/inverse_burgers1d_mlp_models.py \
                --model $model \
                --dir burgers_inverse-endpoint-$noise_level-noise \
                --seed $seed \
                --device cuda:0 \
                --physics_weight 1 \
                --data_weight 10 \
                --ic_weight 10 \
                --data_points $data_points \
                --lambda_0 $lambda_1 \
                --nu_0 $lambda_2 \
                --nu $true_nu \
                --steps 5000 \
                --noise_level $noise_level \
                --grid_factor 4 \
                --data_regime endpoint

                ((seed++))

            done < "$PARAM_VALUES"
        done
    done
done