import numpy as np

reference = np.array(
    [
        -1.10105761e-13,
        -5.96979630e-14,
        -5.64883618e-14,
        -5.40874097e-14,
        -5.20278135e-14,
        -4.47971181e-14,
        -4.36590662e-14,
        -3.83034369e-14,
        -3.68408380e-14,
        -2.82399237e-14,
        -2.26961027e-14,
        -2.19628725e-14,
        -1.66925866e-14,
        -1.49543387e-14,
        -1.18474682e-14,
        -9.97695695e-15,
        -9.82478234e-15,
        -8.57024236e-15,
        -6.63819634e-15,
        -6.39556202e-15,
        -6.17402436e-15,
        -5.64157061e-15,
        -5.45520300e-15,
        -5.38594269e-15,
        -4.76625162e-15,
        -4.74905106e-15,
        -4.58488685e-15,
        -4.28804384e-15,
        -4.18716578e-15,
        -4.07005602e-15,
        -4.03356559e-15,
        -3.76470330e-15,
        -3.76011434e-15,
        -3.48929360e-15,
        -3.48584661e-15,
        -3.39831582e-15,
        -3.23721528e-15,
        -3.18086072e-15,
        -3.09428541e-15,
        -3.07491056e-15,
        -3.02959209e-15,
        -2.97822949e-15,
        -2.93801970e-15,
        -2.82642023e-15,
        -2.81571424e-15,
        -2.80015145e-15,
        -2.72934583e-15,
        -2.66901419e-15,
        -2.60215526e-15,
        -2.54789634e-15,
        -2.47477372e-15,
        -2.44248297e-15,
        -2.34977317e-15,
        -2.32474280e-15,
        -2.25942247e-15,
        -2.14760064e-15,
        -2.13374735e-15,
        -2.09467547e-15,
        -1.99142897e-15,
        -1.98381759e-15,
        -1.92628655e-15,
        -1.71825276e-15,
        -1.70806972e-15,
        -1.69509303e-15,
        -1.67091053e-15,
        -1.63311452e-15,
        -1.60267222e-15,
        -1.59771281e-15,
        -1.57953887e-15,
        -1.54206542e-15,
        -1.48588857e-15,
        -1.47811218e-15,
        -1.41320321e-15,
        -1.36790012e-15,
        -1.36574619e-15,
        -1.35209353e-15,
        -1.26579174e-15,
        -1.21982044e-15,
        -1.18725198e-15,
        -1.12398191e-15,
        -1.12144378e-15,
        -1.07825109e-15,
        -1.06188848e-15,
        -1.02299443e-15,
        -1.02036549e-15,
        -9.85712141e-16,
        -9.24427073e-16,
        -8.75732185e-16,
        -8.56518370e-16,
        -8.55467130e-16,
        -7.57749641e-16,
        -6.85725320e-16,
        -6.73127729e-16,
        -6.34678911e-16,
        -6.05148150e-16,
        -5.25565808e-16,
        -4.79680552e-16,
        -4.59254820e-16,
        -4.45909226e-16,
        -4.19566986e-16,
        -4.19385975e-16,
        -4.13047764e-16,
        -3.66977754e-16,
        -3.34353674e-16,
        -3.33889591e-16,
        -2.66938940e-16,
        -2.17540224e-16,
        -1.04579827e-16,
        -9.04291866e-17,
        -6.42096007e-17,
        -5.31697387e-17,
        -1.87600758e-17,
        8.18569232e-17,
        1.09353058e-16,
        1.97331412e-16,
        2.14134383e-16,
        2.48497584e-16,
        2.51946077e-16,
        2.71795049e-16,
        3.29023301e-16,
        3.59942039e-16,
        3.66926532e-16,
        4.28196843e-16,
        4.44478485e-16,
        4.72905960e-16,
        5.02278485e-16,
        5.90192911e-16,
        6.38572567e-16,
        6.51480924e-16,
        6.84543688e-16,
        7.04899744e-16,
        7.05426707e-16,
        7.17128866e-16,
        7.66388644e-16,
        7.97135555e-16,
        8.03976416e-16,
        8.40864717e-16,
        9.24787379e-16,
        9.97209841e-16,
        1.02445560e-15,
        1.04592482e-15,
        1.05556756e-15,
        1.08151144e-15,
        1.18104100e-15,
        1.24677932e-15,
        1.30016718e-15,
        1.33557970e-15,
        1.34928061e-15,
        1.48182932e-15,
        1.59543126e-15,
        1.62281278e-15,
        1.63096823e-15,
        1.68056973e-15,
        1.68219400e-15,
        1.72091264e-15,
        1.83428203e-15,
        1.87703579e-15,
        1.93725189e-15,
        1.97765270e-15,
        1.97905816e-15,
        2.04304433e-15,
        2.11818970e-15,
        2.18834690e-15,
        2.22941732e-15,
        2.25925970e-15,
        2.27483384e-15,
        2.32728154e-15,
        2.33099289e-15,
        2.39463055e-15,
        2.48279246e-15,
        2.52257211e-15,
        2.60892645e-15,
        2.66052681e-15,
        2.75246260e-15,
        2.86651844e-15,
        2.86747740e-15,
        2.92115439e-15,
        2.99110619e-15,
        3.11652812e-15,
        3.31540278e-15,
        3.34673371e-15,
        3.54071313e-15,
        3.56014412e-15,
        3.67162221e-15,
        3.71657009e-15,
        3.86377880e-15,
        4.13716287e-15,
        4.46443492e-15,
        4.69177075e-15,
        4.88955930e-15,
        5.22790261e-15,
        5.53492731e-15,
        6.28987961e-15,
        6.65771109e-15,
        7.86891050e-15,
        8.17274245e-15,
        8.90467376e-15,
        9.89302046e-15,
        1.47235409e-14,
        2.04249590e-14,
        2.14532049e-14,
        2.37543157e-14,
        2.52717140e-14,
        2.87578458e-14,
        3.21341876e-14,
        3.64146327e-14,
        3.65551761e-14,
        4.08981478e-14,
        4.39670917e-14,
        5.44376162e-14,
        8.92273058e-14,
        2.12000000e02,
    ]
)

reference2 = np.array(
    [
        -1.10105761e-13,
        -5.96979630e-14,
        -5.64883618e-14,
        -5.40874097e-14,
        -5.20278135e-14,
        -4.47971181e-14,
        -4.36590662e-14,
        -3.83034369e-14,
        -3.68408380e-14,
        -2.82399237e-14,
        -2.26961027e-14,
        -2.19628725e-14,
        -1.66925866e-14,
        -1.49543387e-14,
        -1.18474682e-14,
        -9.97695695e-15,
        -9.82478234e-15,
        -8.57024236e-15,
        -6.63819634e-15,
        -6.39556202e-15,
        -6.17402436e-15,
        -5.64157061e-15,
        -5.45520300e-15,
        -5.38594269e-15,
        -4.76625162e-15,
        -4.74905106e-15,
        -4.58488685e-15,
        -4.28804384e-15,
        -4.18716578e-15,
        -4.07005602e-15,
        -4.03356559e-15,
        -3.76470330e-15,
        -3.76011434e-15,
        -3.48929360e-15,
        -3.48584661e-15,
        -3.39831582e-15,
        -3.23721528e-15,
        -3.18086072e-15,
        -3.09428541e-15,
        -3.07491056e-15,
        -3.02959209e-15,
        -2.97822949e-15,
        -2.93801970e-15,
        -2.82642023e-15,
        -2.81571424e-15,
        -2.80015145e-15,
        -2.72934583e-15,
        -2.66901419e-15,
        -2.60215526e-15,
        -2.54789634e-15,
        -2.47477372e-15,
        -2.44248297e-15,
        -2.34977317e-15,
        -2.32474280e-15,
        -2.25942247e-15,
        -2.14760064e-15,
        -2.13374735e-15,
        -2.09467547e-15,
        -1.99142897e-15,
        -1.98381759e-15,
        -1.92628655e-15,
        -1.71825276e-15,
        -1.70806972e-15,
        -1.69509303e-15,
        -1.67091053e-15,
        -1.63311452e-15,
        -1.60267222e-15,
        -1.59771281e-15,
        -1.57953887e-15,
        -1.54206542e-15,
        -1.48588857e-15,
        -1.47811218e-15,
        -1.41320321e-15,
        -1.36790012e-15,
        -1.36574619e-15,
        -1.35209353e-15,
        -1.26579174e-15,
        -1.21982044e-15,
        -1.18725198e-15,
        -1.12398191e-15,
        -1.12144378e-15,
        -1.07825109e-15,
        -1.06188848e-15,
        -1.02299443e-15,
        -1.02036549e-15,
        -9.85712141e-16,
        -9.24427073e-16,
        -8.75732185e-16,
        -8.56518370e-16,
        -8.55467130e-16,
        -7.57749641e-16,
        -6.85725320e-16,
        -6.73127729e-16,
        -6.34678911e-16,
        -6.05148150e-16,
        -5.25565808e-16,
        -4.79680552e-16,
        -4.59254820e-16,
        -4.45909226e-16,
        -4.19566986e-16,
        -4.19385975e-16,
        -4.13047764e-16,
        -3.66977754e-16,
        -3.34353674e-16,
        -3.33889591e-16,
        -2.66938940e-16,
        -2.17540224e-16,
        -1.04579827e-16,
        -9.04291866e-17,
        -6.42096007e-17,
        -5.31697387e-17,
        -1.87600758e-17,
        8.18569232e-17,
        1.09353058e-16,
        1.97331412e-16,
        2.14134383e-16,
        2.48497584e-16,
        2.51946077e-16,
        2.71795049e-16,
        3.29023301e-16,
        3.59942039e-16,
        3.66926532e-16,
        4.28196843e-16,
        4.44478485e-16,
        4.72905960e-16,
        5.02278485e-16,
        5.90192911e-16,
        6.38572567e-16,
        6.51480924e-16,
        6.84543688e-16,
        7.04899744e-16,
        7.05426707e-16,
        7.17128866e-16,
        7.66388644e-16,
        7.97135555e-16,
        8.03976416e-16,
        8.40864717e-16,
        9.24787379e-16,
        9.97209841e-16,
        1.02445560e-15,
        1.04592482e-15,
        1.05556756e-15,
        1.08151144e-15,
        1.18104100e-15,
        1.24677932e-15,
        1.30016718e-15,
        1.33557970e-15,
        1.34928061e-15,
        1.48182932e-15,
        1.59543126e-15,
        1.62281278e-15,
        1.63096823e-15,
        1.68056973e-15,
        1.68219400e-15,
        1.72091264e-15,
        1.83428203e-15,
        1.87703579e-15,
        1.93725189e-15,
        1.97765270e-15,
        1.97905816e-15,
        2.04304433e-15,
        2.11818970e-15,
        2.18834690e-15,
        2.22941732e-15,
        2.25925970e-15,
        2.27483384e-15,
        2.32728154e-15,
        2.33099289e-15,
        2.39463055e-15,
        2.48279246e-15,
        2.52257211e-15,
        2.60892645e-15,
        2.66052681e-15,
        2.75246260e-15,
        2.86651844e-15,
        2.86747740e-15,
        2.92115439e-15,
        2.99110619e-15,
        3.11652812e-15,
        3.31540278e-15,
        3.34673371e-15,
        3.54071313e-15,
        3.56014412e-15,
        3.67162221e-15,
        3.71657009e-15,
        3.86377880e-15,
        4.13716287e-15,
        4.46443492e-15,
        4.69177075e-15,
        4.88955930e-15,
        5.22790261e-15,
        5.53492731e-15,
        6.28987961e-15,
        6.65771109e-15,
        7.86891050e-15,
        8.17274245e-15,
        8.90467376e-15,
        9.89302046e-15,
        1.47235409e-14,
        2.04249590e-14,
        2.14532049e-14,
        2.37543157e-14,
        2.52717140e-14,
        2.87578458e-14,
        3.21341876e-14,
        3.64146327e-14,
        3.65551761e-14,
        4.08981478e-14,
        4.39670917e-14,
        5.44376262e-14,
        8.92273058e-14,
        2.12000000e02,
    ]
)
print(np.sum(reference))
print(np.testing.assert_array_almost_equal_nulp(reference2, reference, nulp=6))
