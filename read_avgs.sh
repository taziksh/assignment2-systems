for f in prof_*.nsys-rep; do
echo "=== $f ==="
nsys stats --report nvtx_pushpop_sum "$f" 2>/dev/null \
	| grep -E 'forward pass|backward pass|optimizer|measured'
done
