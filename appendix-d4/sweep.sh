for s in 30 40 50 60; do
  echo "=== sep=$s ==="
  python toy_cosmochrony_1d_a.py --sep $s --pin --pin_mode asym \
    --kappa_pin 0.2 --kappa_pin2 0.2 \
    --steps 30000 --lr 0.02 --k_eigs 4 --no_plots
done