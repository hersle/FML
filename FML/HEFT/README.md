Clone:
```sh
git clone https://github.com/hersle/FML
cd FML
git checkout gui
cd FML/HEFT
```

Run:
```sh
# First modify 2 file paths in pofk_cross_heft_all_ZA.cpp, if necessary. Then:
make
mpirun -n 32 ./pofk_cross_heft_all_ZA
```

Plot:
```sh
echo '
import matplotlib.pyplot as plt
import numpy as np

with open("pofk_advected_4x4.txt", "r") as f:
    cols = f.readline().strip().split()[1:]
    data = np.loadtxt(f)
    data = {col: data[:, i] for i, col in enumerate(cols)} # dict with column name -> column values
    kcol = cols[0]
    for col in cols[1:]:
        plt.plot(np.log10(data[kcol]), np.log10(data[col]), label = col)
    plt.xlabel(f"log10({kcol})")
    plt.ylabel("log10(P/(Mpc/h)^3)")
    plt.legend()
    plt.savefig("plot.png")
' | python3
```
