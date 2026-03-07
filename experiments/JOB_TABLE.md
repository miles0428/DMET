# N=6, N=7 正式測試 Job 表

## 參數配置
- **U = 10**
- **Fragment Layout**: center=6, edge=4, other=4 (6,4,4)
- **Solver**: CPU / GPU (with fallback)
- **Single-shot**: False (self-consistent)
- **PBC**: False

## 電子數範圍
| N_cells | L (sites) | Total Orbitals | Ne 範圍 |
|---------|-----------|----------------|---------|
| 6       | 13        | 26             | 0 ~ 26 |
| 7       | 15        | 30             | 0 ~ 30 |

## 測試矩陣

### N=6 (L=13, orbitals=26)
| Config Name | t1 | t2 | Ne Count | Total Jobs |
|-------------|----|----|----------|------------|
| N6_t05_t15  | 0.5| 1.5| 27 (0-26)| 27         |
| N6_t15_t05  | 1.5| 0.5| 27 (0-26)| 27         |

### N=7 (L=15, orbitals=30)
| Config Name | t1 | t2 | Ne Count | Total Jobs |
|-------------|----|----|----------|------------|
| N7_t05_t15  | 0.5| 1.5| 31 (0-30)| 31         |
| N7_t15_t05  | 1.5| 0.5| 31 (0-30)| 31         |

## 總計
| 版本   | Configs | Jobs per Config | Total Jobs |
|--------|---------|-----------------|------------|
| CPU    | 4       | 27+27+31+31     | **116**    |
| GPU    | 4       | 27+27+31+31     | **116**    |

## 輸出檔案
- **HDF5**: `DMET/fortest/gpu/{cpu/gpu正式測試}/{Config}_Ne{Ne}.h5`
- **Log**: `DMET/fortest/gpu/{cpu/gpu正式測試}/{Config}_Ne{Ne}.log` (失敗時)
- **Summary**: `DMET/fortest/gpu/{cpu/gpu正式測試}/results_summary.csv`

## 執行命令
```bash
# CPU 版本
cd DMET
LD_PRELOAD=/home/ubuntu/miniforge3/lib/libgomp.so.1 \
  /home/ubuntu/miniforge3/bin/python experiments/test_formal_N67_cpu.py

# GPU 版本 (正式部署用 SSH 到 NTU-server)
ssh NTU-server "source ~/DMET/.venv/bin/activate && cd ~/DMET && python examples/test_formal_N67_gpu.py"
```

## GPU 測試結果 (NTU-server)
已測試 N=6, Ne=10, t1=0.5, t2=1.5, U=10, layout=6,4,4:
- **Solver**: GPU ✓
- **HF energy**: -5.825145
- **DMET energy**: -8.100222
- **收斂 mu**: 0.733245
- **電子數**: 10.00000 ✓

## 預期結果
- 每個 Ne 產生一個獨立的 HDF5 檔案
- 其中一個 Ne 失敗不會影響其他 Ne
- 失敗時會產生 .log 檔案記錄錯誤
