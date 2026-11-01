#!/usr/bin/env bash
# Download the real datasets of the experiments into data/ (or $TUBAL_DATA),
# where src/tubal/data.py reads them. Their terms do not allow redistribution,
# so they are not in this repository (README, "Data"). Every file is checked
# against the SHA-256 of the copy the experiments used, and a mismatch stops
# the script. Files already verified are kept, so a rerun only resumes.
#
#   bash scripts/get_data.sh [videos] [cave] [dcmall]      default: all three
#
#   videos   six QCIF sequences of the Xiph.org test media      72 MB
#   cave     six scenes of the CAVE multispectral database      78 MB
#   dcmall   HYDICE Washington DC Mall                         110 MB, 560 MB on disk
#
# Needs curl and Python 3 ($PYTHON, default python3). dcmall also converts
# dc.tif to dc.npy, with uv if it is installed and otherwise with $PYTHON,
# which then needs numpy and tifffile.

set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${TUBAL_DATA:-data}
PY=${PYTHON:-python3}
UA="Mozilla/5.0 (X11; Linux x86_64)"

sha256() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }

fetch() {  # fetch <url> <file> <bytes> <sha256>
    local url=$1 file=$2 bytes=$3 sum=$4 tries=0
    mkdir -p "$(dirname "$file")"
    while [[ ! -f $file ]] || (( $(wc -c <"$file") < bytes )); do
        (( tries++ < 60 )) || { echo "error: could not download $url" >&2; exit 1; }
        # -C - resumes: the Purdue server resets long transfers
        curl -fsSL -A "$UA" -C - --max-time 300 -o "$file" "$url" || sleep 2
    done
    if [[ $(sha256 "$file") != "$sum" ]]; then
        echo "error: $file is not the file the experiments used; delete it and rerun" >&2
        exit 1
    fi
    echo "ok  $file"
}

videos() {
    local name bytes sum
    while read -r name bytes sum; do
        fetch "https://media.xiph.org/video/derf/y4m/${name}_qcif.y4m" \
              "$DATA/video/${name}_qcif.y4m" "$bytes" "$sum" </dev/null
    done <<'EOF'
akiyo         11406644  df88d83cbf6d99f3ec41f2c1fd2e67665d2a71ff8caa08f8b6bc46bf4567ea2e
carphone      14524448  bd176aee36dd839dcec88671acc811c61f72ea45ee458afb82b35e8b92328aa9
coastguard    11406644  94b615bcb5f89edc601b0e570a599ef584e066e647647b9b5d8b70fb8ebb0146
foreman       11406644  9b37e95ae2d06b3b173d6130965f450009216084bae12b2025248814baf057af
hall_monitor  11406644  f19165140cde8b300cb6ea78c9250b3506665228f7519d843bfcff6269e83051
news          11406644  98e4d83bc8db885eaf52eabeb5aa8c4f6d891c16037541970dce098753167ae4
EOF
}

cave() {
    local name bytes sum
    while read -r name bytes sum; do
        fetch "https://cave.cs.columbia.edu/old/databases/multispectral/zip/${name}_ms.zip" \
              "$DATA/raw/${name}_ms.zip" "$bytes" "$sum" </dev/null
        "$PY" -m zipfile -e "$DATA/raw/${name}_ms.zip" "$DATA/cave/" </dev/null
    done <<'EOF'
balloons      12532814  0ac1bb5aff3327db58ba72d527b1a44d2fa01a6c3f5ef0769d33857301e7af73
beads         14522325  a3aa8767657082f4318d29be3702027507bbc9e98fb57ba9959f051111ea9a89
feathers      13103466  c0201abdbed95008aea4d29fb18a1f466319e8e4592883a914326c7ceee1d5cb
flowers       12228490  ef369ee7e2ec8eb872cde2663cb697fd8d861fb5d784c5459f69ce94116c342f
stuffed_toys  12375041  29973d1169dc55d37679d5fa262c524e977c152512b8cce86060589120dad25d
superballs    13194505  0683034a657d6b09f3adf31e2a2cb8c04d1c136f62e37f2cf4ac3d79179e8f6c
EOF
}

dcmall() {
    local zip=$DATA/raw/Hyperspectral_Project.zip npy=$DATA/dcmall/dc.npy
    local sum=5195a6c5925d08398ff6415861bb751fbe3360bf1801dfd48eca24d7f57d3b20
    fetch https://engineering.purdue.edu/~biehl/Hyperspectral_Project.zip "$zip" 109578046 \
          19c3210145afffc3293a2d62a3af43741323f9be867b80c20abdd1afa2e629a4
    if [[ ! -f $npy || $(sha256 "$npy") != "$sum" ]]; then
        "$PY" -m zipfile -e "$zip" "$DATA/dcmall/"
        # dc.tif holds 191 bands of 1280 x 307 int16; the code reads 1280 x 307 x 191 float32
        local code='import sys, numpy as np, tifffile
np.save(sys.argv[2], np.moveaxis(tifffile.imread(sys.argv[1]), 0, 2).astype(np.float32))'
        local tif=$DATA/dcmall/Hyperspectral_Project/dc.tif
        if command -v uv >/dev/null; then
            uv run --no-project --with numpy --with tifffile python -c "$code" "$tif" "$npy"
        else
            "$PY" -c "$code" "$tif" "$npy"
        fi
        if [[ $(sha256 "$npy") != "$sum" ]]; then
            echo "error: $npy is not the array the experiments used" >&2
            exit 1
        fi
    fi
    echo "ok  $npy"
}

(( $# )) || set -- videos cave dcmall
for what in "$@"; do
    case $what in
        videos | cave | dcmall) "$what" ;;
        *) echo "usage: bash scripts/get_data.sh [videos] [cave] [dcmall]" >&2; exit 2 ;;
    esac
done
