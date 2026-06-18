for size in small medium; do
for ctx in 256 512 1024; do
echo "--- size=$size ctx=$ctx ---"
uv run python -m cs336_systems.benchmark \
	--size $size --mode forward \
	--model.context-length $ctx \
	--batch-size 1 \
	--warmup-steps 5 --measure-steps 5
done; done
