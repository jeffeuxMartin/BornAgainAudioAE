
# region
BSZ = 64
LR = 2e-4
WORKERS = 8
TOTAL = -1
EPOCHS = 100
FeatBSZ = 2048
verbose = False
ModelName = "MyAutoencoder()"

import sys
exec(sys.argv[1] if len(sys.argv) > 1 else '')
print(f"{BSZ = }"
' 'f"{LR = };"
' 'f"{WORKERS = };"
' 'f"{ModelName = };"
# ' 'f"{FeatBSZ = }"
)
print()
# endregion
