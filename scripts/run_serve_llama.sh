#!/usr/bin/env bash
# Launch the tensor-ash server with the GC-rooted Vulkan userspace selected.
set -euo pipefail
cd "$(dirname "$0")/.."

VULKAN_PROFILE=${SIGHURT_VULKAN_PROFILE:-"$HOME/.local/state/nix/profiles/sighurt-vulkan"}
LOADER="$VULKAN_PROFILE/lib/libvulkan.so.1"
NOUVEAU_ICD="$VULKAN_PROFILE/share/vulkan/icd.d/nouveau_icd.x86_64.json"
RADV_ICD="$VULKAN_PROFILE/share/vulkan/icd.d/radeon_icd.x86_64.json"

test -r "$LOADER" || {
    echo "missing Vulkan loader in $VULKAN_PROFILE" >&2
    exit 1
}
test -r "$NOUVEAU_ICD" || {
    echo "missing NVK ICD in $VULKAN_PROFILE" >&2
    exit 1
}
test -r "$RADV_ICD" || {
    echo "missing RADV ICD in $VULKAN_PROFILE" >&2
    exit 1
}

export LD_LIBRARY_PATH="$VULKAN_PROFILE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-"$NOUVEAU_ICD:$RADV_ICD"}
export ML_DEVICE=${ML_DEVICE:-discrete}

exec target/release/serve_llama "$@"
