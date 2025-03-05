#!/bin/bash

# Set the output tar file name
TAR_FILE="videos_archive.tar"

# Set the compression variable (1-22)
COMPRESSION_LEVEL=22

# Compress videos
tar -cvaf ${TAR_FILE}.zst --use-compress-program="zstd --ultra -${COMPRESSION_LEVEL}" videos

echo "Videos successfully archived and compressed into $TAR_FILE.zst"