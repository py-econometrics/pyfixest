import pandas as pd
import matplotlib.pyplot as plt

def panelview(df, unit, time, treat, subsamp = None, t = ""):
  treatment_quilt = df.pivot(index = unit, columns = time, values = treat)
  if subsamp:
    plt.matshow(treatment_quilt.sample(subsamp))
  else:
    plt.matshow(treatment_quilt)
  plt.xlabel(time)
  plt.ylabel(unit)
  plt.title(f"Panel view of {t}")
