import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA

keys = ["front", "back", "left", "right", "space"]
K = len(keys)
#df = pd.read_csv("key_counts.csv")

from models.cgan_dp_keys_tanh import gan_model, G, data_loader

df = pd.DataFrame()

gan_model.load_model()
G.eval()

row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    counts = G.generate(game_state)
    ri = 0
    for row in counts:
        dict = {'robot_mode': game_state[ri][0].item()}
        for i in range(K):
            y = row[i].item() #1 if row[i].item() > 0 else 0
            dict[keys[i]] = y
        df = df.append(dict, ignore_index=True)
        ri += 1
        # print(dict)
    if batch_i % 10 == 0:
        print(batch_i)
    # if batch_i == 100:
    #     break

print("Computing")

maov = MANOVA.from_formula('front + back + left + right + space ~ robot_mode', data=df)
print(maov.mv_test())


"""
                     Multivariate linear model
===================================================================
                                                                   
-------------------------------------------------------------------
       Intercept        Value  Num DF    Den DF    F Value   Pr > F
-------------------------------------------------------------------
          Wilks' lambda 0.7439 5.0000 274414.0000 18891.4610 0.0000
         Pillai's trace 0.2561 5.0000 274414.0000 18891.4610 0.0000
 Hotelling-Lawley trace 0.3442 5.0000 274414.0000 18891.4610 0.0000
    Roy's greatest root 0.3442 5.0000 274414.0000 18891.4610 0.0000
-------------------------------------------------------------------
                                                                   
-------------------------------------------------------------------
        robot_mode       Value  Num DF    Den DF    F Value  Pr > F
-------------------------------------------------------------------
           Wilks' lambda 0.9324 5.0000 274414.0000 3981.9838 0.0000
          Pillai's trace 0.0676 5.0000 274414.0000 3981.9838 0.0000
  Hotelling-Lawley trace 0.0726 5.0000 274414.0000 3981.9838 0.0000
     Roy's greatest root 0.0726 5.0000 274414.0000 3981.9838 0.0000
===================================================================

"""


"""
for original values
                      Multivariate linear model
======================================================================
                                                                      
----------------------------------------------------------------------
       Intercept         Value  Num DF    Den DF     F Value    Pr > F
----------------------------------------------------------------------
          Wilks' lambda  0.0201 5.0000 274414.0000 2673670.2837 0.0000
         Pillai's trace  0.9799 5.0000 274414.0000 2673670.2837 0.0000
 Hotelling-Lawley trace 48.7160 5.0000 274414.0000 2673670.2837 0.0000
    Roy's greatest root 48.7160 5.0000 274414.0000 2673670.2837 0.0000
----------------------------------------------------------------------
                                                                      
----------------------------------------------------------------------
           robot_mode       Value  Num DF    Den DF    F Value  Pr > F
----------------------------------------------------------------------
              Wilks' lambda 0.9168 5.0000 274414.0000 4980.3789 0.0000
             Pillai's trace 0.0832 5.0000 274414.0000 4980.3789 0.0000
     Hotelling-Lawley trace 0.0907 5.0000 274414.0000 4980.3789 0.0000
        Roy's greatest root 0.0907 5.0000 274414.0000 4980.3789 0.0000
======================================================================

"""
