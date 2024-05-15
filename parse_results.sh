docker \
    run \
    --rm \
    -v $(pwd):/workspaces/pvae \
    -w /workspaces/pvae \
    -u $(id -u):$(id -g) \
    pvae:1.3-cuda10.1-cudnn7-runtime
python -u parse_results.py
