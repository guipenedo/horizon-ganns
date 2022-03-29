import pandas as pd
from statsmodels.multivariate.manova import MANOVA

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

from models.cgan_dp_keys_tanh import gan_model, G, data_loader

df = pd.DataFrame()

gan_model.load_model()
G.eval()


def add_rows(keys_d, state, df, generator=1):
    ri = 0
    for row in keys_d:
        dict = {'generator': generator, 'robot_mode': state[ri][0].item()}
        for i in range(K):
            y = 1 if row[i].item() > 0 else 0
            dict[keys[i]] = y
        df = df.append(dict, ignore_index=True)
        ri += 1
    return df


row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    counts = G.generate(game_state)

    df = add_rows(counts, game_state, df, generator=1)

    df = add_rows(real_keys, game_state, df, generator=0)

    if batch_i % 10 == 0:
        print(batch_i)
#     if batch_i == 3:
#         break
#
# print(df)

print("Computing")

maov = MANOVA.from_formula('front + back + left + right + space ~ generator', data=df)
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


"""
generator mode comparison
                      Multivariate linear model
======================================================================
                                                                      
----------------------------------------------------------------------
       Intercept         Value  Num DF    Den DF     F Value    Pr > F
----------------------------------------------------------------------
          Wilks' lambda  0.0433 5.0000 548834.0000 2426422.8863 0.0000
         Pillai's trace  0.9567 5.0000 548834.0000 2426422.8863 0.0000
 Hotelling-Lawley trace 22.1053 5.0000 548834.0000 2426422.8863 0.0000
    Roy's greatest root 22.1053 5.0000 548834.0000 2426422.8863 0.0000
----------------------------------------------------------------------
                                                                      
----------------------------------------------------------------------
           generator        Value  Num DF    Den DF    F Value  Pr > F
----------------------------------------------------------------------
              Wilks' lambda 0.9336 5.0000 548834.0000 7809.7982 0.0000
             Pillai's trace 0.0664 5.0000 548834.0000 7809.7982 0.0000
     Hotelling-Lawley trace 0.0711 5.0000 548834.0000 7809.7982 0.0000
        Roy's greatest root 0.0711 5.0000 548834.0000 7809.7982 0.0000
======================================================================

"""
