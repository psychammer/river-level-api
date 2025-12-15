# prediction_api.py
"""
Water Level Forecasting API
Deploys the trained STGAE-GAT-Transformer model for real-time predictions
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta
import math
import logging
import uvicorn
import os
import re 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PRECOMPUTED_METRICS = {
    "full_model": {
        "MONTALBAN": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [3.5501, 3.5952, 3.7058, 3.5881, 3.5595, 3.5994, 3.6028, 3.6399, 3.5864, 3.4449, 3.6069, 3.6443, 3.7519, 3.7076, 3.6795, 3.5213, 3.5310, 3.6784, 3.4603, 3.7107, 3.5676, 3.5606, 3.5523, 3.5105, 3.5643, 3.4433, 3.4655, 3.5159, 3.6108, 3.4238, 3.4807, 3.5974, 3.4222, 3.4508, 3.4063, 3.4854, 3.4275, 3.4760, 3.5078, 3.4034, 3.4286, 3.5664, 3.4194, 3.4522, 3.3953, 3.5171, 3.5012, 3.4510],
            "rmse": [4.1568, 4.2081, 4.2700, 4.1900, 4.1623, 4.2245, 4.2375, 4.2148, 4.1747, 4.0299, 4.2192, 4.2572, 4.3503, 4.3188, 4.2886, 4.0939, 4.0940, 4.2750, 4.0763, 4.3437, 4.1655, 4.1876, 4.1591, 4.1043, 4.1752, 4.0437, 4.0467, 4.1297, 4.2247, 4.0241, 4.0395, 4.2054, 4.0145, 4.0438, 4.0097, 4.0929, 4.0374, 4.0693, 4.0810, 3.9643, 4.0200, 4.1508, 3.9965, 4.0699, 3.9933, 4.1203, 4.0889, 4.0704],
            "nse":  [-18.5936, -19.0795, -19.6751, -18.9078, -18.6452, -19.2365, -19.3615, -19.1437, -18.7619, -17.4148, -19.1854, -19.5514, -20.4598, -20.1505, -19.8554, -18.0052, -18.0056, -19.7235, -17.8413, -20.3943, -18.6757, -18.8847, -18.6154, -18.1018, -18.7672, -17.5421, -17.5694, -18.3386, -19.2388, -17.3622, -17.5032, -19.0546, -17.2751, -17.5431, -17.2312, -17.9959, -17.4840, -17.7777, -17.8855, -16.8204, -17.3247, -18.5371, -17.1112, -17.7825, -17.0826, -18.2512, -17.9588, -17.7878]
        },
        "NANGKA": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [0.2969, 0.2519, 0.2559, 0.2633, 0.2268, 0.2763, 0.2601, 0.2322, 0.2835, 0.2705, 0.2261, 0.2651, 0.2737, 0.2747, 0.2822, 0.2225, 0.2146, 0.1853, 0.2649, 0.2707, 0.2745, 0.2954, 0.2611, 0.2797, 0.2720, 0.2729, 0.2088, 0.2949, 0.2705, 0.2727, 0.2588, 0.2658, 0.2964, 0.2638, 0.3116, 0.2763, 0.2632, 0.2820, 0.2295, 0.2884, 0.3192, 0.2342, 0.2498, 0.1990, 0.2950, 0.2761, 0.2846, 0.2882],
            "rmse": [0.5957, 0.5850, 0.6069, 0.6164, 0.5963, 0.6499, 0.6362, 0.6162, 0.6577, 0.6498, 0.6083, 0.6319, 0.6257, 0.6549, 0.6607, 0.6184, 0.5964, 0.5934, 0.6572, 0.6451, 0.6722, 0.6871, 0.6612, 0.6606, 0.6668, 0.6497, 0.6145, 0.6833, 0.6772, 0.6474, 0.6631, 0.6746, 0.6878, 0.6667, 0.7148, 0.6710, 0.6667, 0.6753, 0.6417, 0.6865, 0.7372, 0.6539, 0.6417, 0.6241, 0.6994, 0.6680, 0.6794, 0.6817],
            "nse":  [-0.3207, -0.2735, -0.3708, -0.4142, -0.3234, -0.5717, -0.5065, -0.4130, -0.6101, -0.5712, -0.3773, -0.4859, -0.4569, -0.5963, -0.6248, -0.4235, -0.3239, -0.3106, -0.6075, -0.5490, -0.6819, -0.7569, -0.6271, -0.6242, -0.6548, -0.5711, -0.4052, -0.7378, -0.7065, -0.5600, -0.6366, -0.6935, -0.7604, -0.6542, -0.9018, -0.6759, -0.6545, -0.6974, -0.5324, -0.7542, -1.0224, -0.5912, -0.5326, -0.4495, -0.8204, -0.6607, -0.7178, -0.7293]
        },
        "SAN MATEO": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [1.1216, 1.1066, 1.0837, 1.1675, 1.1752, 1.1012, 1.1394, 1.1792, 1.2153, 1.1310, 1.1440, 1.1974, 1.1344, 1.2115, 1.1924, 1.1122, 1.1416, 1.1706, 1.1422, 1.1660, 1.1933, 1.1732, 1.1768, 1.1760, 1.1833, 1.2050, 1.2274, 1.2421, 1.1928, 1.2612, 1.2995, 1.2958, 1.2818, 1.2254, 1.2345, 1.2424, 1.2793, 1.2880, 1.2746, 1.2793, 1.2689, 1.3176, 1.2765, 1.2465, 1.2443, 1.2526, 1.2824, 1.1957],
            "rmse": [1.5137, 1.5113, 1.5071, 1.5613, 1.5703, 1.5132, 1.5697, 1.6083, 1.6156, 1.5688, 1.5948, 1.6326, 1.6136, 1.6606, 1.6463, 1.6159, 1.6373, 1.6596, 1.6419, 1.6661, 1.6813, 1.6818, 1.6872, 1.6880, 1.7051, 1.7306, 1.7313, 1.7428, 1.7229, 1.7827, 1.7979, 1.8049, 1.8132, 1.7631, 1.7760, 1.7911, 1.8161, 1.8314, 1.8440, 1.8216, 1.8252, 1.8661, 1.8366, 1.8178, 1.8242, 1.8329, 1.8614, 1.8183],
            "nse":  [0.8369, 0.8375, 0.8384, 0.8265, 0.8245, 0.8371, 0.8247, 0.8159, 0.8142, 0.8249, 0.8190, 0.8103, 0.8147, 0.8038, 0.8071, 0.8142, 0.8092, 0.8040, 0.8082, 0.8025, 0.7988, 0.7987, 0.7974, 0.7972, 0.7931, 0.7869, 0.7867, 0.7838, 0.7887, 0.7738, 0.7700, 0.7682, 0.7660, 0.7788, 0.7755, 0.7717, 0.7653, 0.7613, 0.7580, 0.7639, 0.7629, 0.7522, 0.7600, 0.7648, 0.7632, 0.7609, 0.7534, 0.7647]
        },
        "STO NINO": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [1.6610, 1.6674, 1.7131, 1.6824, 1.6801, 1.7614, 1.6561, 1.6562, 1.6771, 1.6123, 1.6677, 1.6630, 1.6783, 1.7206, 1.6649, 1.7147, 1.6534, 1.6557, 1.6186, 1.7312, 1.7296, 1.7168, 1.6793, 1.7395, 1.6799, 1.6591, 1.7121, 1.6900, 1.6815, 1.6682, 1.6373, 1.6325, 1.5928, 1.6066, 1.5956, 1.5891, 1.6386, 1.6723, 1.6692, 1.6764, 1.5987, 1.6186, 1.6696, 1.6737, 1.6578, 1.6143, 1.5977, 1.6877],
            "rmse": [1.9694, 1.9572, 1.9912, 1.9613, 1.9686, 2.0348, 1.9533, 1.9449, 1.9577, 1.8975, 1.9393, 1.9453, 1.9525, 1.9828, 1.9604, 1.9949, 1.9521, 1.9402, 1.9154, 2.0183, 2.0088, 1.9944, 1.9699, 2.0209, 1.9716, 1.9489, 1.9896, 1.9606, 1.9514, 1.9495, 1.9262, 1.9156, 1.8864, 1.8823, 1.8866, 1.8702, 1.9031, 1.9345, 1.9492, 1.9658, 1.8871, 1.9136, 1.9435, 1.9414, 1.9361, 1.8986, 1.8928, 1.9605],
            "nse":  [-0.6663, -0.6457, -0.7033, -0.6525, -0.6649, -0.7788, -0.6390, -0.6250, -0.6465, -0.5469, -0.6156, -0.6257, -0.6377, -0.6890, -0.6510, -0.7096, -0.6371, -0.6172, -0.5760, -0.7499, -0.7335, -0.7087, -0.6671, -0.7544, -0.6699, -0.6317, -0.7006, -0.6514, -0.6360, -0.6327, -0.5938, -0.5765, -0.5287, -0.5221, -0.5290, -0.5026, -0.5560, -0.6077, -0.6322, -0.6602, -0.5299, -0.5732, -0.6227, -0.6191, -0.6103, -0.5486, -0.5391, -0.6513]
        },
        "TUMANA": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [1.6829, 1.6716, 1.7247, 1.6895, 1.6361, 1.7085, 1.6164, 1.6781, 1.6920, 1.6974, 1.6734, 1.6324, 1.6947, 1.6834, 1.6624, 1.7638, 1.6762, 1.7012, 1.6626, 1.6796, 1.6940, 1.7046, 1.6908, 1.7253, 1.7018, 1.6577, 1.7010, 1.7013, 1.6748, 1.6908, 1.6401, 1.6851, 1.5895, 1.6405, 1.6168, 1.5712, 1.6122, 1.6817, 1.6610, 1.7099, 1.6063, 1.6652, 1.6341, 1.6604, 1.6452, 1.6272, 1.6480, 1.6908],
            "rmse": [1.9145, 1.8969, 1.9356, 1.8911, 1.8620, 1.9086, 1.8372, 1.8731, 1.8852, 1.8950, 1.8730, 1.8466, 1.8847, 1.8755, 1.8596, 1.9486, 1.8786, 1.8820, 1.8596, 1.8821, 1.8935, 1.8999, 1.8766, 1.9208, 1.8959, 1.8485, 1.8923, 1.8834, 1.8676, 1.8748, 1.8386, 1.8693, 1.7901, 1.8286, 1.8021, 1.7665, 1.8108, 1.8650, 1.8437, 1.8892, 1.7931, 1.8524, 1.8237, 1.8426, 1.8245, 1.8163, 1.8365, 1.8789],
            "nse":  [-1.1820, -1.1420, -1.2304, -1.1289, -1.0639, -1.1685, -1.0094, -1.0887, -1.1157, -1.1376, -1.0884, -1.0300, -1.1145, -1.0940, -1.0586, -1.2603, -1.1007, -1.1085, -1.0587, -1.1086, -1.1342, -1.1488, -1.0963, -1.1962, -1.1397, -1.0341, -1.1316, -1.1116, -1.0763, -1.0923, -1.0122, -1.0800, -0.9076, -0.9904, -0.9333, -0.8575, -0.9518, -1.0704, -1.0235, -1.1246, -0.9140, -1.0425, -0.9799, -1.0209, -0.9815, -0.9637, -1.0077, -1.1013]
        }
    },
    "ablated_model": {
        "MONTALBAN": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [0.5345, 0.6201, 0.5324, 0.6536, 0.5096, 0.4923, 0.5522, 0.5412, 0.6291, 0.5375, 0.5611, 0.5087, 0.5041, 0.5773, 0.6372, 0.6084, 0.6160, 0.5786, 0.6550, 0.5840, 0.5899, 0.6003, 0.6503, 0.6523, 0.6146, 0.5804, 0.6692, 0.6488, 0.5897, 0.6343, 0.5995, 0.6465, 0.6224, 0.6539, 0.6569, 0.6634, 0.7423, 0.6435, 0.5889, 0.7396, 0.6181, 0.6358, 0.5954, 0.6899, 0.6699, 0.6455, 0.6839, 0.6563],
            "rmse": [0.7274, 0.8039, 0.7411, 0.8558, 0.7612, 0.7564, 0.8182, 0.8118, 0.8629, 0.8201, 0.8049, 0.7990, 0.8401, 0.9261, 0.9359, 0.8945, 0.9685, 0.9408, 0.9915, 0.9364, 0.9530, 0.9513, 0.9790, 0.9964, 1.0200, 0.9617, 1.0725, 0.9852, 1.0098, 1.0751, 1.0478, 1.1050, 1.0253, 1.1157, 1.0990, 1.1177, 1.1956, 1.1217, 1.0558, 1.2329, 1.1359, 1.1296, 1.0843, 1.2169, 1.1933, 1.1088, 1.1858, 1.1188],
            "nse":  [0.4001, 0.2673, 0.3772, 0.1696, 0.3429, 0.3513, 0.2408, 0.2527, 0.1556, 0.2374, 0.2653, 0.2761, 0.1997, 0.0276, 0.0068, 0.0927, -0.0636, -0.0037, -0.1147, 0.0056, -0.0298, -0.0261, -0.0867, -0.1257, -0.1798, -0.0488, -0.3043, -0.1005, -0.1563, -0.3106, -0.2450, -0.3846, -0.1921, -0.4116, -0.3697, -0.4165, -0.6210, -0.4267, -0.2640, -0.7237, -0.4631, -0.4469, -0.3331, -0.6793, -0.6146, -0.3942, -0.5945, -0.4194]
        },
        "NANGKA": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [0.2142, 0.2895, 0.2292, 0.3396, 0.2340, 0.2374, 0.2885, 0.3033, 0.2916, 0.2749, 0.3713, 0.2934, 0.2666, 0.2887, 0.2853, 0.3226, 0.3302, 0.3333, 0.4008, 0.3277, 0.3121, 0.2868, 0.3972, 0.3484, 0.3653, 0.3697, 0.3734, 0.3794, 0.3794, 0.3825, 0.3470, 0.4504, 0.4336, 0.3420, 0.4255, 0.3605, 0.3522, 0.3320, 0.3314, 0.3437, 0.3481, 0.3883, 0.4051, 0.3253, 0.3268, 0.3368, 0.3856, 0.4181],
            "rmse": [0.5456, 0.5865, 0.5412, 0.6350, 0.5829, 0.5935, 0.6259, 0.6475, 0.6354, 0.6191, 0.6707, 0.6358, 0.6117, 0.6391, 0.6435, 0.6554, 0.6765, 0.7045, 0.7214, 0.6829, 0.6868, 0.6515, 0.7351, 0.7064, 0.7330, 0.7145, 0.7257, 0.7206, 0.7515, 0.7503, 0.7434, 0.8371, 0.7767, 0.7285, 0.7757, 0.7456, 0.7729, 0.7485, 0.7367, 0.7847, 0.7730, 0.7969, 0.8103, 0.7915, 0.7833, 0.7233, 0.8016, 0.7716],
            "nse":  [-0.1077, -0.2802, -0.0902, -0.5006, -0.2646, -0.3107, -0.4578, -0.5603, -0.5025, -0.4263, -0.6744, -0.5045, -0.3927, -0.5200, -0.5409, -0.5987, -0.7034, -0.8474, -0.9367, -0.7355, -0.7554, -0.5797, -1.0109, -0.8569, -0.9994, -0.9002, -0.9600, -0.9326, -1.1016, -1.0949, -1.0567, -1.6079, -1.2450, -0.9754, -1.2393, -1.0688, -1.2231, -1.0853, -1.0200, -1.2915, -1.2240, -1.3638, -1.4436, -1.3313, -1.2835, -0.9472, -1.3916, -1.2157]
        },
        "SAN MATEO": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [1.1244, 1.2014, 1.1207, 1.1415, 1.0653, 1.0447, 1.1743, 1.1634, 1.1236, 1.1488, 1.0267, 1.0841, 1.0523, 1.0773, 1.0717, 1.0753, 1.0652, 1.0400, 1.0841, 1.1078, 1.1247, 1.1380, 1.1206, 1.0792, 1.1473, 1.1343, 1.1603, 1.1454, 1.1356, 1.1118, 1.2543, 1.2932, 1.1719, 1.2300, 1.2261, 1.1917, 1.1158, 1.1434, 1.1608, 1.1055, 1.1934, 1.1487, 1.1342, 1.1863, 1.1577, 1.1904, 1.2988, 1.2166],
            "rmse": [1.5577, 1.6442, 1.5827, 1.5853, 1.5918, 1.5350, 1.5916, 1.6095, 1.6095, 1.6372, 1.6019, 1.5391, 1.5716, 1.6370, 1.6270, 1.6175, 1.6383, 1.6516, 1.6419, 1.6320, 1.7141, 1.6716, 1.7576, 1.6590, 1.6794, 1.6947, 1.6986, 1.6770, 1.6420, 1.6522, 1.7515, 1.7827, 1.7767, 1.8352, 1.7628, 1.7754, 1.6950, 1.7948, 1.7670, 1.7811, 1.7898, 1.7695, 1.7994, 1.8261, 1.8247, 1.8073, 1.8621, 1.8358],
            "nse":  [0.8273, 0.8076, 0.8217, 0.8211, 0.8197, 0.8323, 0.8197, 0.8157, 0.8157, 0.8092, 0.8174, 0.8314, 0.8242, 0.8093, 0.8116, 0.8138, 0.8090, 0.8059, 0.8081, 0.8105, 0.7909, 0.8011, 0.7802, 0.8041, 0.7993, 0.7956, 0.7947, 0.7999, 0.8081, 0.8057, 0.7817, 0.7738, 0.7754, 0.7603, 0.7788, 0.7757, 0.7955, 0.7707, 0.7778, 0.7742, 0.7720, 0.7772, 0.7696, 0.7627, 0.7630, 0.7675, 0.7532, 0.7602]
        },
        "STO NINO": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [0.7224, 0.8178, 0.7937, 0.7027, 0.6938, 0.6830, 0.6968, 0.6793, 0.7371, 0.7053, 0.7156, 0.7687, 0.7677, 0.7235, 0.7595, 0.7085, 0.7762, 0.7666, 0.7604, 0.8530, 0.8399, 0.8434, 0.8165, 0.8553, 0.9440, 0.8175, 0.8335, 0.9176, 0.8653, 0.9215, 0.9416, 0.9872, 0.8809, 0.9049, 0.9218, 0.9027, 1.0046, 0.9505, 1.0133, 0.9219, 1.0320, 1.0993, 0.9184, 0.9490, 1.0188, 0.9904, 0.9983, 0.9899],
            "rmse": [1.1485, 1.1986, 1.1853, 1.1437, 1.1406, 1.1410, 1.1516, 1.1495, 1.1690, 1.1594, 1.1621, 1.1881, 1.1981, 1.1723, 1.1844, 1.1578, 1.1944, 1.1906, 1.1888, 1.2395, 1.2277, 1.2295, 1.2082, 1.2347, 1.2861, 1.2141, 1.2238, 1.2728, 1.2507, 1.2796, 1.2894, 1.3227, 1.2627, 1.2791, 1.2952, 1.2734, 1.3379, 1.3101, 1.3410, 1.2924, 1.3596, 1.4062, 1.3000, 1.3207, 1.3514, 1.3464, 1.3438, 1.3441],
            "nse":  [0.4333, 0.3829, 0.3964, 0.4381, 0.4411, 0.4407, 0.4302, 0.4324, 0.4130, 0.4225, 0.4199, 0.3936, 0.3834, 0.4096, 0.3974, 0.4242, 0.3872, 0.3910, 0.3929, 0.3400, 0.3525, 0.3506, 0.3729, 0.3451, 0.2894, 0.3667, 0.3566, 0.3040, 0.3280, 0.2966, 0.2858, 0.2484, 0.3151, 0.2971, 0.2794, 0.3034, 0.2311, 0.2626, 0.2274, 0.2824, 0.2059, 0.1506, 0.2740, 0.2506, 0.2154, 0.2212, 0.2242, 0.2239]
        },
        "TUMANA": {
            "horizons": ["1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "25h", "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h", "40h", "41h", "42h", "43h", "44h", "45h", "46h", "47h", "48h"],
            "mae":  [0.7914, 0.8382, 0.8074, 0.8354, 0.8750, 0.8080, 0.8718, 0.8558, 0.8252, 0.7473, 0.9291, 0.8695, 0.8594, 0.8202, 0.8508, 0.8538, 0.8818, 0.9321, 0.9244, 0.9088, 0.9227, 0.9418, 0.8846, 0.9743, 1.0032, 0.8893, 0.8760, 0.8953, 1.0005, 1.0244, 0.9160, 1.0180, 0.8782, 0.9348, 1.0461, 1.0094, 1.0881, 1.0506, 1.0509, 0.9604, 1.0256, 1.1031, 1.0529, 1.0319, 1.1204, 0.9913, 1.0522, 1.0957],
            "rmse": [0.9439, 0.9896, 0.9523, 0.9882, 1.0245, 0.9633, 1.0133, 1.0062, 0.9724, 0.9280, 1.0796, 1.0107, 1.0181, 0.9792, 1.0002, 1.0126, 1.0357, 1.0755, 1.0714, 1.0644, 1.0666, 1.0829, 1.0401, 1.1210, 1.1367, 1.0689, 1.0403, 1.0706, 1.1500, 1.1675, 1.0784, 1.1542, 1.0584, 1.0941, 1.2014, 1.1814, 1.2561, 1.2017, 1.2049, 1.1395, 1.1841, 1.2486, 1.2131, 1.1926, 1.2668, 1.1630, 1.2065, 1.2419],
            "nse":  [0.4696, 0.4171, 0.4601, 0.4187, 0.3752, 0.4476, 0.3887, 0.3973, 0.4371, 0.4873, 0.3061, 0.3919, 0.3829, 0.4292, 0.4044, 0.3896, 0.3615, 0.3114, 0.3167, 0.3256, 0.3227, 0.3020, 0.3560, 0.2519, 0.2308, 0.3199, 0.3557, 0.3177, 0.2128, 0.1887, 0.3077, 0.2070, 0.3332, 0.2874, 0.1408, 0.1692, 0.0609, 0.1404, 0.1358, 0.2271, 0.1654, 0.0719, 0.1240, 0.1534, 0.0448, 0.1949, 0.1335, 0.0819]
        }
    }
}



# --- Configuration ---
class Config:
    # Model paths (adjust these to your deployed environment)
    MODEL_DIR = './models/'
    STGAE_MODEL_PATH = MODEL_DIR + 'stgae_full_model.pth'
    ENCODER_PATH = MODEL_DIR + 'stgae_encoder.pth'
    FULL_MODEL_PATH = MODEL_DIR + 'full_forecaster_model.pth'
    ABLATED_MODEL_PATH = MODEL_DIR + 'ablated_forecaster_model.pth'  # Added ablated model
    ADJ_MATRIX_PATH = MODEL_DIR + 'adjacency_matrix.npy'
    SCALER_MEAN_PATH = MODEL_DIR + 'scaler_mean.npy'
    SCALER_STD_PATH = MODEL_DIR + 'scaler_std.npy'
    
    # Model specifications
    NUM_STATIONS = 5
    NUM_FEATURES = 10
    TARGET_FEATURE_IDX = 0  # water level
    LOOKBACK_WINDOW = 24 * 3  # 3 days
    FORECAST_HORIZONS = list(range(1, 49))
    
    # Station configuration
    STATION_CONFIG = {
        'MONTALBAN': {
            'coords': (14.733083, 121.130580),
            'obscd': '11102202'
        },
        'NANGKA': {
            'coords': (14.674022, 121.109319),
            'obscd': '11103202'
        },
        'SAN MATEO': {
            'coords': (14.679547, 121.109733),
            'obscd': '11104202'
        },
        'STO NINO': {
            'coords': (14.635941, 121.093122),
            'obscd': '11105202'
        },
        'TUMANA': {
            'coords': (14.656427, 121.096508),
            'obscd': '11106202'
        }
    }
    
    STATION_ORDER = ['MONTALBAN', 'NANGKA', 'SAN MATEO', 'STO NINO', 'TUMANA']
    FEATURE_ORDER = [
        'waterlevel',           
        'temperature_2m',        
        'relative_humidity_2m',  
        'dew_point_2m',         
        'rain',        
        'wind_speed_10m',       
        'surface_pressure',     
        'wind_gusts_10m',       
        'wind_direction_10m',   
        'cloud_cover'           
    ]
    
    # Model hyperparameters (must match training)
    STGAE_GCN_HIDDEN = 64
    STGAE_GRU_HIDDEN_FACTOR = 64
    GAT_IN_FEATURES = STGAE_GCN_HIDDEN
    GAT_HIDDEN_DIM = 32
    GAT_HEADS = 4
    TRANSFORMER_D_MODEL = GAT_HIDDEN_DIM * GAT_HEADS
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_FF_DIM = 256
    
    # API endpoints
    WATER_API_BASE = "http://121.58.193.173:8080/water/map_list.do"
    WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# --- Model Definitions (from your training code) ---

class STGAE(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        
        self.decoder_gru = nn.GRU(gru_hidden, gru_hidden, batch_first=True)
        self.decoder_fc = nn.Linear(gru_hidden, gcn_hidden * config.NUM_STATIONS)
        self.decoder_gcn = GCNConv(gcn_hidden, in_features)
        
        self.relu = nn.Tanh()

    def forward(self, x, edge_index):
        x_original = x.clone()
        batch_size, seq_len, num_nodes, _ = x.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            xt_reshaped = xt.reshape(-1, x.size(3))
            
            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )

            gcn_out = self.relu(self.encoder_gcn(xt_reshaped, edge_index_batched))
            gcn_out_reshaped = gcn_out.reshape(batch_size, num_nodes, -1)
            gcn_outputs.append(gcn_out_reshaped)
            
        gcn_sequence = torch.stack(gcn_outputs, dim=1)
        gru_in = gcn_sequence.reshape(batch_size, seq_len, -1)
        _, latent_rep = self.encoder_gru(gru_in) 
        latent_rep = latent_rep.squeeze(0)

        decoder_gru_in = latent_rep.unsqueeze(1).repeat(1, seq_len, 1)
        gru_out, _ = self.decoder_gru(decoder_gru_in)
        fc_out = self.relu(self.decoder_fc(gru_out))
        
        decoder_outputs = []
        for t in range(seq_len):
            fct = fc_out[:, t, :]
            fct_reshaped = fct.reshape(-1, self.encoder_gcn.out_channels)

            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )

            recon_t = self.decoder_gcn(fct_reshaped, edge_index_batched)
            recon_t_reshaped = recon_t.reshape(batch_size, num_nodes, -1)
            decoder_outputs.append(recon_t_reshaped)
            
        reconstruction = torch.stack(decoder_outputs, dim=1)
        reconstruction = reconstruction + x_original
        return reconstruction

class FrozenSTGAEEncoder(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden, encoder_weights_path):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        self.activation = nn.Tanh()
        self.load_encoder_weights(encoder_weights_path)
        for param in self.parameters():
            param.requires_grad = False
    
    def load_encoder_weights(self, weights_path):
        encoder_state = torch.load(weights_path, map_location=config.DEVICE, weights_only=True)
        self.encoder_gcn.load_state_dict(encoder_state['encoder_gcn'])
        self.encoder_gru.load_state_dict(encoder_state['encoder_gru'])
    
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            xt_reshaped = xt.reshape(-1, x.size(3))
            
            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )
            
            gcn_out = self.activation(self.encoder_gcn(xt_reshaped, edge_index_batched))
            gcn_out_reshaped = gcn_out.reshape(batch_size, num_nodes, -1)
            gcn_outputs.append(gcn_out_reshaped)
        
        gcn_features = torch.stack(gcn_outputs, dim=1)
        gru_in = gcn_features.reshape(batch_size, seq_len, -1)
        _, temporal_encoding = self.encoder_gru(gru_in)
        temporal_encoding = temporal_encoding.squeeze(0)
        
        return gcn_features, temporal_encoding

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, heads):
        super().__init__()
        self.gat = GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim * heads)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ff_dim, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        output = self.transformer(x)
        return output

class MultiHorizonForecastHead(nn.Module):
    def __init__(self, input_dim, horizons, num_stations, target_feature_idx):
        super().__init__()
        self.horizons = horizons
        self.num_stations = num_stations
        self.target_feature_idx = target_feature_idx
        
        self.forecast_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, num_stations)
            )
            for h in horizons
        })
        
    def forward(self, x):
        predictions = {}
        for horizon in self.horizons:
            pred = self.forecast_heads[f'horizon_{horizon}'](x)
            predictions[horizon] = pred
        return predictions

class STGAEGATTransformer(nn.Module):
    def __init__(self, config, encoder_weights_path):
        super().__init__()
        
        self.stgae_encoder = FrozenSTGAEEncoder(
            in_features=config.NUM_FEATURES,
            gcn_hidden=config.STGAE_GCN_HIDDEN,
            gru_hidden=config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS,
            encoder_weights_path=encoder_weights_path
        )
        
        self.gat = GraphAttentionLayer(
            in_features=config.GAT_IN_FEATURES,
            hidden_dim=config.GAT_HIDDEN_DIM,
            heads=config.GAT_HEADS
        )
        
        self.transformer = TemporalTransformer(
            d_model=config.TRANSFORMER_D_MODEL,
            n_heads=config.TRANSFORMER_HEADS,
            n_layers=config.TRANSFORMER_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            max_seq_len=config.LOOKBACK_WINDOW
        )
        
        self.forecast_head = MultiHorizonForecastHead(
            input_dim=config.TRANSFORMER_D_MODEL,
            horizons=config.FORECAST_HORIZONS,
            num_stations=config.NUM_STATIONS,
            target_feature_idx=config.TARGET_FEATURE_IDX
        )
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        with torch.no_grad():
            gcn_features, temporal_encoding = self.stgae_encoder(x, edge_index)
        
        gcn_flat = gcn_features.reshape(-1, gcn_features.size(-1))
        
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len):
            edge_index_batched = torch.cat(
                [edge_index_batched, edge_index + i * num_nodes], dim=1
            )
        
        gat_out = self.gat(gcn_flat, edge_index_batched)
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        transformer_input = gat_features.mean(dim=2)
        transformer_out = self.transformer(transformer_input)
        last_timestep = transformer_out[:, -1, :]
        predictions = self.forecast_head(last_timestep)
        
        return predictions

class AblatedGATTransformer(nn.Module):
    """
    Ablated model: Raw Features → GAT → Transformer → Multi-Horizon Forecasting
    (No STGAE preprocessing)
    """
    def __init__(self, config):
        super().__init__()
        
        # GAT takes raw features directly (NUM_FEATURES instead of STGAE_GCN_HIDDEN)
        self.gat = GraphAttentionLayer(
            in_features=config.NUM_FEATURES,  # Direct raw features
            hidden_dim=config.GAT_HIDDEN_DIM,
            heads=config.GAT_HEADS
        )
        
        self.transformer = TemporalTransformer(
            d_model=config.TRANSFORMER_D_MODEL,
            n_heads=config.TRANSFORMER_HEADS,
            n_layers=config.TRANSFORMER_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            max_seq_len=config.LOOKBACK_WINDOW
        )
        
        self.forecast_head = MultiHorizonForecastHead(
            input_dim=config.TRANSFORMER_D_MODEL,
            horizons=config.FORECAST_HORIZONS,
            num_stations=config.NUM_STATIONS,
            target_feature_idx=config.TARGET_FEATURE_IDX
        )
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Apply GAT directly to raw features
        x_flat = x.reshape(-1, num_features)
        
        # Create batched edge index for GAT
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len):
            edge_index_batched = torch.cat(
                [edge_index_batched, edge_index + i * num_nodes], dim=1
            )
        
        # GAT forward (processes raw features)
        gat_out = self.gat(x_flat, edge_index_batched)
        
        # Reshape back
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        
        # Aggregate spatial features for transformer input
        transformer_input = gat_features.mean(dim=2)
        
        # Temporal modeling with Transformer
        transformer_out = self.transformer(transformer_input)
        
        # Use the last timestep for forecasting
        last_timestep = transformer_out[:, -1, :]
        
        # Multi-horizon forecasting
        predictions = self.forecast_head(last_timestep)
        
        return predictions

# --- Data Fetching and Processing ---

class DataFetcher:
    """Fetches and processes real-time data from APIs"""
    
    @staticmethod
    def fetch_water_levels(start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        """Fetch water level data for all stations"""
        logger.info(f"Fetching water levels from {start_datetime} to {end_datetime}")
        
        all_data = []
        current = start_datetime
        
        while current <= end_datetime:
            ymdhm = current.strftime('%Y%m%d%H%M')
            url = f"{config.WATER_API_BASE}?ymdhm={ymdhm}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for station in data:
                    obscd = station.get('obscd')
                    station_name = None
                    for name, cfg in config.STATION_CONFIG.items():
                        if cfg['obscd'] == obscd:
                            station_name = name
                            break
                    
                    if station_name and station.get('wl'):
                        try:
                            # 1. Convert the raw value to a string to be safe.
                            wl_string = str(station['wl'])
                            
                            # 2. Use the regex to strip out anything that isn't a digit or a decimal point.
                            #    This handles '21.62(*)', '26 (*)', '[Error]', etc.
                            cleaned_wl_string = re.sub(r'[^\d.]', '', wl_string)
                            
                            # 3. If the cleaned string is not empty, convert it to a float.
                            if cleaned_wl_string:
                                wl_value = float(cleaned_wl_string)
                                all_data.append({
                                    'datetime': current,
                                    'station': station_name,
                                    'waterlevel': wl_value
                                })
                        except (ValueError, TypeError):
                            # This is a final safety net. If conversion still fails,
                            # just skip this data point.
                            pass
                            
            except Exception as e:
                logger.warning(f"Failed to fetch water data for {ymdhm}: {e}")
            
            current += timedelta(hours=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='datetime',
                columns='station',
                values='waterlevel'
            )
            return df_pivot
        else:
            return pd.DataFrame()
    
    @staticmethod
    def fetch_weather_data(station_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data for a specific station"""
        coords = config.STATION_CONFIG[station_name]['coords']
        
        params = {
            'latitude': coords[0],
            'longitude': coords[1],
            'hourly': ','.join([
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'rain', 'wind_speed_10m', 'wind_gusts_10m', 'cloud_cover',
                'wind_direction_10m', 'surface_pressure'
            ]),
            'timezone': 'Asia/Singapore',
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            response = requests.get(config.WEATHER_API_BASE, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Rename columns to match feature order
            df.rename(columns={'rain': 'rain'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch weather for {station_name}: {e}")
            return pd.DataFrame()

class DataProcessor:
    """Processes raw data for model input"""
    
    def __init__(self):
        # Load scalers and adjacency matrix
        self.mean = np.load(config.SCALER_MEAN_PATH)
        self.std = np.load(config.SCALER_STD_PATH)
        self.adj_matrix = np.load(config.ADJ_MATRIX_PATH)
        self.edge_index = torch.tensor(
            np.array(np.where(self.adj_matrix > 0)), 
            dtype=torch.long
        ).to(config.DEVICE)
        
        # Load STGAE for imputation
        self.stgae_model = STGAE(
            in_features=config.NUM_FEATURES,
            gcn_hidden=config.STGAE_GCN_HIDDEN,
            gru_hidden=config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS
        ).to(config.DEVICE)
        
        stgae_state = torch.load(config.STGAE_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
        self.stgae_model.load_state_dict(stgae_state)
        self.stgae_model.eval()
    
    def prepare_features(self, water_df: pd.DataFrame, weather_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Combine water and weather data into feature array"""
        
        # Create a complete time index
        time_index = water_df.index
        
        # Initialize feature array
        num_timesteps = len(time_index)
        feature_array = np.full(
            (num_timesteps, config.NUM_STATIONS, config.NUM_FEATURES),
            np.nan,
            dtype=np.float32
        )
        
        # Fill in water levels
        for s_idx, station in enumerate(config.STATION_ORDER):
            if station in water_df.columns:
                feature_array[:, s_idx, 0] = water_df[station].values
        
        # Fill in weather features
        for s_idx, station in enumerate(config.STATION_ORDER):
            if station in weather_data and not weather_data[station].empty:
                weather_df = weather_data[station].reindex(time_index)
                
                for f_idx, feature in enumerate(config.FEATURE_ORDER[1:], start=1):
                    if feature in weather_df.columns:
                        feature_array[:, s_idx, f_idx] = weather_df[feature].values
        
        return feature_array
    
    def impute_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Use STGAE to impute missing values"""
        if features.shape[0] < config.LOOKBACK_WINDOW:
            # Pad with NaNs if not enough data
            pad_size = config.LOOKBACK_WINDOW - features.shape[0]
            padding = np.full((pad_size, features.shape[1], features.shape[2]), np.nan)
            features = np.vstack([padding, features])
        
        # Scale features
        scaled_features = (features - self.mean) / (self.std + 1e-8)
        
        # Prepare input tensor
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        input_clean = torch.nan_to_num(input_tensor)
        
        # Run STGAE for imputation
        with torch.no_grad():
            reconstruction = self.stgae_model(input_clean, self.edge_index)
        
        # Fill NaN values with reconstructed values
        reconstruction_np = reconstruction.squeeze(0).cpu().numpy()
        scaled_features_imputed = scaled_features.copy()
        nan_mask = np.isnan(scaled_features)
        scaled_features_imputed[nan_mask] = reconstruction_np[nan_mask]
        
        # Inverse scale
        features_imputed = (scaled_features_imputed * (self.std + 1e-8)) + self.mean
        
        # Return only the lookback window
        return features_imputed[-config.LOOKBACK_WINDOW:]

# --- Model Manager ---

class ModelManager:
    """Manages both full and ablated forecasting models"""
    
    def __init__(self):
        # Load both models
        self.full_model = STGAEGATTransformer(config, config.ENCODER_PATH).to(config.DEVICE)
        self.ablated_model = AblatedGATTransformer(config).to(config.DEVICE)
        
        # Load trained weights for full model
        try:
            full_state_dict = torch.load(config.FULL_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
            self.full_model.load_state_dict(full_state_dict)
            self.full_model.eval()
            logger.info("Full model (with STGAE) loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Full model not found at {config.FULL_MODEL_PATH}")
            self.full_model = None
        
        # Load trained weights for ablated model
        try:
            ablated_state_dict = torch.load(config.ABLATED_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
            self.ablated_model.load_state_dict(ablated_state_dict)
            self.ablated_model.eval()
            logger.info("Ablated model (without STGAE) loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Ablated model not found at {config.ABLATED_MODEL_PATH}")
            self.ablated_model = None
        
        self.processor = DataProcessor()
        
        # Check which models are available
        self.available_models = []
        if self.full_model is not None:
            self.available_models.append("full")
        if self.ablated_model is not None:
            self.available_models.append("ablated")
        
        if not self.available_models:
            raise RuntimeError("No models available! Please ensure model files exist.")
        
        logger.info(f"Available models: {self.available_models}")
    
    def predict(self, features: np.ndarray, model_type: str = "full") -> Dict[int, np.ndarray]:
        """
        Make predictions for all forecast horizons
        
        Args:
            features: Input feature array
            model_type: "full" for STGAE-GAT-Transformer, "ablated" for GAT-Transformer only
        """
        
        # Validate model type
        if model_type not in self.available_models:
            raise ValueError(f"Model type '{model_type}' not available. Choose from: {self.available_models}")
        
        # Impute missing values
        features_imputed = self.processor.impute_missing_values(features)
        
        # Scale features
        features_scaled = (features_imputed - self.processor.mean) / (self.processor.std + 1e-8)
        
        # Prepare tensor
        input_tensor = torch.tensor(
            features_scaled, 
            dtype=torch.float32
        ).unsqueeze(0).to(config.DEVICE)
        
        # Select model
        model = self.full_model if model_type == "full" else self.ablated_model
        
        # Make predictions
        with torch.no_grad():
            predictions_scaled = model(
                torch.nan_to_num(input_tensor),
                self.processor.edge_index
            )
        
        # Inverse scale predictions (only for water level)
        predictions = {}
        water_mean = self.processor.mean[0]
        water_std = self.processor.std[0]
        
        for horizon in config.FORECAST_HORIZONS:
            pred_scaled = predictions_scaled[horizon].cpu().numpy()
            pred_original = (pred_scaled * water_std) + water_mean
            predictions[horizon] = pred_original.squeeze()
        
        return predictions

# --- API Definition ---

app = FastAPI(title="Water Level Forecasting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_manager
    model_manager = ModelManager()
    logger.info("API started successfully")

# Request/Response models
class PredictionRequest(BaseModel):
    lookback_hours: Optional[int] = 72  # Default to 3 days
    model_type: Optional[str] = "full"  # "full" or "ablated"

class StationPrediction(BaseModel):
    station: str
    current_level: Optional[float]
    predictions: Dict[str, float]
    alert_levels: Dict[str, float]

class PredictionResponse(BaseModel):
    timestamp: str
    model_type: str
    data_range: Dict[str, str]
    predictions: List[StationPrediction]

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online", 
        "models": {
            "available": model_manager.available_models if model_manager else [],
            "full": "STGAE-GAT-Transformer",
            "ablated": "GAT-Transformer (no STGAE)"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_water_levels(request: PredictionRequest):
    """
    Predict water levels for all stations at multiple horizons
    
    Args:
        lookback_hours: Number of hours of historical data to use
        model_type: "full" (with STGAE) or "ablated" (without STGAE)
    """
    try:
        # Validate model type
        if request.model_type not in ["full", "ablated"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_type. Choose 'full' or 'ablated'"
            )
        
        if request.model_type not in model_manager.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_type}' not available. Available models: {model_manager.available_models}"
            )
        
        # Calculate time range
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=request.lookback_hours)
        
        logger.info(f"Fetching data from {start_time} to {end_time}")
        logger.info(f"Using model: {request.model_type}")
        
        # Fetch water level data
        fetcher = DataFetcher()
        water_df = fetcher.fetch_water_levels(start_time, end_time)
        
        if water_df.empty:
            raise HTTPException(status_code=404, detail="No water level data available")
        
        # Fetch weather data for all stations
        weather_data = {}
        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')
        
        for station in config.STATION_ORDER:
            weather_df = fetcher.fetch_weather_data(station, start_date, end_date)
            weather_data[station] = weather_df
        
        # Prepare features
        features = model_manager.processor.prepare_features(water_df, weather_data)
        
        # Make predictions with selected model
        predictions_raw = model_manager.predict(features, model_type=request.model_type)
        
        # Format response
        response_data = []
        
        for s_idx, station in enumerate(config.STATION_ORDER):
            # Get current water level
            current_level = None
            if station in water_df.columns:
                last_values = water_df[station].dropna()
                if not last_values.empty:
                    current_level = float(last_values.iloc[-1])
            
            # Format predictions
            station_predictions = {}
            for horizon in config.FORECAST_HORIZONS:
                pred_value = float(predictions_raw[horizon][s_idx])
                station_predictions[f"{horizon}h"] = round(pred_value, 2)
            
            # Get alert levels (these would be from your database/config)
            # You should update these with actual values for each station
            alert_levels = {
                "alert": 22.4 if station == "MONTALBAN" else 16.5,
                "alarm": 23.0 if station == "MONTALBAN" else 17.1,
                "critical": 23.6 if station == "MONTALBAN" else 17.7
            }
            
            response_data.append(StationPrediction(
                station=station,
                current_level=current_level,
                predictions=station_predictions,
                alert_levels=alert_levels
            ))
        
        return PredictionResponse(
            timestamp=datetime.now().isoformat(),
            model_type=request.model_type,
            data_range={
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            predictions=response_data
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stations")
async def get_stations():
    """Get list of available stations"""
    stations = []
    for name, cfg in config.STATION_CONFIG.items():
        stations.append({
            "name": name,
            "coordinates": {
                "lat": cfg['coords'][0],
                "lon": cfg['coords'][1]
            },
            "obscd": cfg['obscd']
        })
    return {"stations": stations}

@app.get("/model_info")
async def get_model_info():
    """Get information about the models, including performance metrics."""
    return {
        "available_models": model_manager.available_models if model_manager else [],
        "models": {
            "full": {
                "name": "STGAE-GAT-Transformer",
                "description": "Full model with STGAE encoder for feature extraction",
                "components": ["STGAE Encoder", "GAT", "Transformer", "Multi-Horizon Head"]
            },
            "ablated": {
                "name": "GAT-Transformer",
                "description": "Ablated model without STGAE preprocessing",
                "components": ["GAT", "Transformer", "Multi-Horizon Head"]
            }
        },
        "features": config.FEATURE_ORDER,
        "forecast_horizons": config.FORECAST_HORIZONS,
        "lookback_window": config.LOOKBACK_WINDOW,
        "num_stations": config.NUM_STATIONS,
        "device": config.DEVICE,
        "performance_metrics": PRECOMPUTED_METRICS #Add the metrics to the response
    }

@app.post("/compare_models")
async def compare_models(request: PredictionRequest):
    """
    Compare predictions from both models
    """
    try:
        if not all(m in model_manager.available_models for m in ["full", "ablated"]):
            raise HTTPException(
                status_code=400,
                detail="Both models must be available for comparison"
            )
        
        # Calculate time range
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=request.lookback_hours)
        
        # Fetch data
        fetcher = DataFetcher()
        water_df = fetcher.fetch_water_levels(start_time, end_time)
        
        if water_df.empty:
            raise HTTPException(status_code=404, detail="No water level data available")
        
        weather_data = {}
        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')
        
        for station in config.STATION_ORDER:
            weather_df = fetcher.fetch_weather_data(station, start_date, end_date)
            weather_data[station] = weather_df
        
        # Prepare features
        features = model_manager.processor.prepare_features(water_df, weather_data)
        
        # Get predictions from both models
        full_predictions = model_manager.predict(features, model_type="full")
        ablated_predictions = model_manager.predict(features, model_type="ablated")
        
        # Format comparison
        comparison = {}
        for s_idx, station in enumerate(config.STATION_ORDER):
            station_comparison = {}
            for horizon in config.FORECAST_HORIZONS:
                full_pred = float(full_predictions[horizon][s_idx])
                ablated_pred = float(ablated_predictions[horizon][s_idx])
                station_comparison[f"{horizon}h"] = {
                    "full_model": round(full_pred, 2),
                    "ablated_model": round(ablated_pred, 2),
                    "difference": round(full_pred - ablated_pred, 2),
                    "percent_diff": round(((full_pred - ablated_pred) / ablated_pred * 100) if ablated_pred != 0 else 0, 1)
                }
            comparison[station] = station_comparison
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)