#!/bin/bash
# Detecta JAVA_HOME en tiempo de ejecución, independiente de la arquitectura
# (amd64, arm64, etc.) y lo exporta antes de arrancar el pipeline.
set -e

JAVA_BIN=$(readlink -f "$(which java)")
export JAVA_HOME=$(dirname $(dirname "$JAVA_BIN"))
export PATH="$JAVA_HOME/bin:$PATH"

echo "[entrypoint] JAVA_HOME detectado: $JAVA_HOME"
exec "$@"
