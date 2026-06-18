for size in small medium; do
for ctx in 256 512 1024; do
for mode in forward full; do
echo "=== $size $ctx $mode ==="
uv run nsys profile \
	-o prof_${size}_ctx${ctx}_${mode} \
	--trace=cuda,nvtx \
	--capture-range=nvtx \
	--nvtx-capture=measured \
	-- python -m cs336_systems.benchmark \
	--size $size --mode $mode \
	--model.context-length $ctx \
	--batch-size 1 \
	--warmup-steps 5 --measure-steps 5
done; done; done
