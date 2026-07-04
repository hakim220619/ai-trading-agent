# 🧺 BasketScalper — EA Basket Scalping (Target Profit)

EA MQL5 native untuk **MetaTrader 5**. Membuka beberapa posisi scalping searah
sinyal (satu **basket**), lalu **menutup SEMUA posisi sekaligus** saat total
floating profit basket mencapai target. **Bukan martingale** — lot tetap, tidak
menambah lot saat rugi.

Dituning untuk **XAUUSD (M1)**, tapi semua parameter bisa diubah.

---

## 📦 Instalasi

1. Buka MetaTrader 5 → **File → Open Data Folder**.
2. Masuk ke `MQL5/Experts/`.
3. Copy `BasketScalper.mq5` ke folder itu.
4. Buka **MetaEditor** (F4) → buka `BasketScalper.mq5` → tekan **Compile** (F7).
   Pastikan **0 error**.
5. Kembali ke MT5, di **Navigator → Expert Advisors** akan muncul `BasketScalper`.
6. Drag ke chart **XAUUSD M1**.
7. Di tab **Common**, centang **Allow Algo Trading**. Pastikan tombol
   **Algo Trading** di toolbar MT5 juga aktif (hijau).

> ⚠️ **Uji dulu di akun DEMO / Strategy Tester** sebelum akun real.

---

## ⚙️ Cara Kerja Singkat

```
Sinyal (EMA cross + RSI filter di M1)
        │
        ▼
Buka posisi (lot tetap / risk %)  ──► spacing antar entry (ATR/points)
        │                              max InpMaxPositions per basket
        ▼
Basket = kumpulan posisi searah
        │
        ├─ Total profit ≥ Target  ──► TUTUP SEMUA  ✅
        ├─ Total loss  ≤ Basket SL ──► TUTUP SEMUA  🛑
        └─ Trailing basket dari puncak profit ──► TUTUP SEMUA  📉
```

Exit **dikelola per-basket**, bukan per-posisi (order tidak memasang SL/TP sendiri).

---

## 🔧 Config Lengkap (Input)

### UMUM
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpMagic` | 220619 | Identitas EA. Beda EA/chart = beda magic. |
| `InpComment` | BasketScalper | Komentar pada order. |
| `InpSlippagePoints` | 20 | Max slippage saat eksekusi (points). |

### SINYAL ENTRY
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpSignalTF` | M1 | Timeframe sinyal. |
| `InpUseMAFilter` | true | Pakai EMA cross sebagai penentu arah. |
| `InpFastEMA` / `InpSlowEMA` | 8 / 21 | Periode EMA cepat & lambat. |
| `InpUseRSIFilter` | true | RSI sebagai filter izin entry. |
| `InpRSIPeriod` | 14 | Periode RSI. |
| `InpRSIBuyMax` | 70 | BUY hanya jika RSI < nilai ini. |
| `InpRSISellMin` | 30 | SELL hanya jika RSI > nilai ini. |
| `InpATRTF` / `InpATRPeriod` | M1 / 14 | ATR untuk jarak spacing adaptif. |

### ARAH & LOT
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpDirectionMode` | DIR_SIGNAL | Ikut sinyal / BUY only / SELL only. |
| `InpLotMode` | LOT_FIXED | Lot tetap atau risk %. |
| `InpFixedLot` | 0.01 | Lot per posisi (jika fixed). |
| `InpRiskPercent` | 1.0 | Risk % balance per posisi (jika risk). |
| `InpRiskSLPoints` | 300 | Jarak SL (points) untuk hitung lot risk. |

### BASKET / SPACING
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpMaxPositions` | 5 | Maksimum posisi dalam 1 basket. |
| `InpUseSpacing` | true | Wajib ada jarak antar entry. |
| `InpStepMode` | STEP_ATR | Jarak antar entry: ATR atau points. |
| `InpStepPoints` | 150 | Jarak antar entry (points) — mode STEP_POINTS. |
| `InpStepATRmult` | 1.0 | Jarak = ATR × nilai ini — mode STEP_ATR. |
| `InpEntryCooldownSec` | 5 | Jeda minimum antar entry (detik). |

### EXIT — TARGET PROFIT
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpTargetMode` | TGT_MONEY | Target dalam uang atau % balance. |
| `InpTargetMoney` | 5.0 | Target profit basket (uang akun). |
| `InpTargetPercent` | 0.5 | Target profit basket (% balance). |

### EXIT — BASKET STOP LOSS
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpUseBasketSL` | true | Aktifkan basket stop loss. |
| `InpSLMode` | SL_MONEY | SL dalam uang atau % balance. |
| `InpBasketSLMoney` | 30.0 | Basket SL (uang akun). |
| `InpBasketSLPercent` | 3.0 | Basket SL (% balance). |

### EXIT — BASKET TRAILING
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpUseTrailing` | true | Trailing pada profit basket. |
| `InpTrailStartMoney` | 3.0 | Mulai trailing saat profit ≥ ini. |
| `InpTrailStepMoney` | 1.5 | Jarak trailing dari puncak profit. |

### FILTER SPREAD & WAKTU
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpMaxSpreadPoints` | 300 | Spread maksimum untuk entry (points). |
| `InpUseTimeFilter` | false | Aktifkan filter jam (server time). |
| `InpStartHour` / `InpEndHour` | 0 / 24 | Rentang jam entry. |
| `InpTradeMonday` / `InpTradeFriday` | true | Izin trading Senin/Jumat. |
| `InpFridayForceClose` | true | Tutup semua sebelum weekend. |
| `InpFridayCloseHour` | 22 | Jam tutup paksa Jumat. |

### BATAS HARIAN
| Input | Default | Keterangan |
|-------|---------|-----------|
| `InpUseDailyLimit` | false | Aktifkan batas profit/loss harian. |
| `InpDailyProfitStop` | 50 | Stop entry jika profit harian ≥ ini. |
| `InpDailyLossStop` | 50 | Stop entry jika loss harian ≥ ini. |

---

## 🎯 Rekomendasi Setting

### XAUUSD — akun kecil (balance ~$100–500), konservatif
```
InpSignalTF        = M1
InpFastEMA         = 8
InpSlowEMA         = 21
InpUseRSIFilter    = true
InpLotMode         = LOT_FIXED
InpFixedLot        = 0.01
InpMaxPositions    = 3
InpStepMode        = STEP_ATR
InpStepATRmult     = 1.0
InpTargetMode      = TGT_MONEY
InpTargetMoney     = 2.0
InpUseBasketSL     = true
InpBasketSLMoney   = 15.0
InpUseTrailing     = true
InpTrailStartMoney = 1.5
InpTrailStepMoney  = 0.8
InpMaxSpreadPoints = 300
```

### XAUUSD — balance ~$1000+, % balance
```
InpLotMode         = LOT_FIXED
InpFixedLot        = 0.02
InpMaxPositions    = 5
InpTargetMode      = TGT_PERCENT
InpTargetPercent   = 0.5
InpUseBasketSL     = true
InpSLMode          = SL_PERCENT
InpBasketSLPercent = 3.0
InpUseDailyLimit   = true
InpDailyProfitStop = 30
InpDailyLossStop   = 30
```

---

## ⚠️ Catatan Risiko

- **Basket tanpa martingale tetap berisiko**: jika harga tren melawan terus,
  basket bisa membesar sampai `InpMaxPositions` dan kena `InpBasketSL`. Selalu
  set basket SL dengan nilai yang sanggup kamu terima.
- **XAUUSD volatile** — spread bisa melebar saat news. Gunakan filter spread &
  pertimbangkan `InpUseTimeFilter` untuk hindari jam news.
- **Selalu backtest di Strategy Tester** dengan data "Every tick based on real
  ticks" sebelum live.
- EA tidak memasang SL/TP di tiap order (by design) — proteksi ada di level
  basket. Jika MT5/VPS mati, tidak ada SL broker. Pastikan koneksi VPS stabil.
