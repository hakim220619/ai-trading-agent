//+------------------------------------------------------------------+
//|                                              BasketScalper.mq5    |
//|            Basket Scalping EA — Target Profit (all-close)         |
//|                                                                  |
//|  Konsep:                                                         |
//|   - Buka beberapa posisi scalping searah sinyal (satu "basket"). |
//|   - Tutup SEMUA posisi sekaligus saat TOTAL floating profit       |
//|     basket mencapai target (uang / % balance / points).           |
//|   - BUKAN martingale: lot tetap, tidak menambah lot saat rugi.    |
//|   - Ada basket Stop-Loss, basket trailing, filter jam & spread.   |
//|                                                                  |
//|  Dituning untuk XAUUSD (M1) tapi semua input bisa diubah.         |
//+------------------------------------------------------------------+
#property copyright "ai-trading-agent"
#property link      "https://github.com/hakim220619/ai-trading-agent"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//==================================================================//
//  INPUT / CONFIG                                                  //
//==================================================================//

//--- Umum -----------------------------------------------------------
input group           "=== UMUM ==="
input long     InpMagic            = 220619;      // Magic number (identitas EA)
input string   InpComment          = "BasketScalper"; // Komentar order
input ulong    InpSlippagePoints   = 20;          // Max slippage (points/deviation)

//--- Sinyal Entry (scalping) ---------------------------------------
input group           "=== SINYAL ENTRY (SCALPING) ==="
input ENUM_TIMEFRAMES InpSignalTF   = PERIOD_M1;   // Timeframe sinyal
input bool     InpUseMAFilter      = true;         // Pakai filter EMA cross
input int      InpFastEMA          = 8;            // Periode EMA cepat
input int      InpSlowEMA          = 21;           // Periode EMA lambat
input bool     InpUseRSIFilter     = true;         // Pakai filter RSI
input int      InpRSIPeriod        = 14;           // Periode RSI
input double   InpRSIBuyMax        = 70.0;         // BUY hanya jika RSI < nilai ini
input double   InpRSISellMin       = 30.0;         // SELL hanya jika RSI > nilai ini
input ENUM_TIMEFRAMES InpATRTF      = PERIOD_M1;   // Timeframe ATR (jarak adaptif)
input int      InpATRPeriod        = 14;           // Periode ATR

//--- Arah & Lot -----------------------------------------------------
input group           "=== ARAH & LOT ==="
enum ENUM_DIR_MODE { DIR_SIGNAL=0, DIR_BUY_ONLY=1, DIR_SELL_ONLY=2 };
input ENUM_DIR_MODE InpDirectionMode = DIR_SIGNAL; // Arah entry
enum ENUM_LOT_MODE { LOT_FIXED=0, LOT_RISK_PERCENT=1 };
input ENUM_LOT_MODE InpLotMode      = LOT_FIXED;   // Mode penentuan lot
input double   InpFixedLot         = 0.01;         // Lot tetap (jika LOT_FIXED)
input double   InpRiskPercent      = 1.0;          // Risk % balance per leg (jika RISK)
input double   InpRiskSLPoints     = 300;          // Jarak SL (points) utk hitung lot risk

//--- Basket / Grid --------------------------------------------------
input group           "=== BASKET / SPACING ==="
input int      InpMaxPositions     = 5;            // Maksimum posisi dalam 1 basket
input bool     InpUseSpacing       = true;         // Wajib ada jarak antar entry
enum ENUM_STEP_MODE { STEP_POINTS=0, STEP_ATR=1 };
input ENUM_STEP_MODE InpStepMode    = STEP_ATR;    // Mode jarak antar entry
input int      InpStepPoints       = 150;          // Jarak antar entry (points) - STEP_POINTS
input double   InpStepATRmult      = 1.0;          // Jarak = ATR x nilai ini - STEP_ATR
input int      InpEntryCooldownSec = 5;            // Jeda minimum antar entry (detik)

//--- Exit: Target Profit Basket ------------------------------------
input group           "=== EXIT: TARGET PROFIT BASKET ==="
enum ENUM_TARGET_MODE { TGT_MONEY=0, TGT_PERCENT=1 };
input ENUM_TARGET_MODE InpTargetMode = TGT_MONEY;  // Mode target profit
input double   InpTargetMoney      = 5.0;          // Target profit basket (uang akun)
input double   InpTargetPercent    = 0.5;          // Target profit basket (% balance)

//--- Exit: Basket Stop Loss ----------------------------------------
input group           "=== EXIT: BASKET STOP LOSS ==="
input bool     InpUseBasketSL      = true;         // Aktifkan basket stop loss
enum ENUM_SL_MODE { SL_MONEY=0, SL_PERCENT=1 };
input ENUM_SL_MODE InpSLMode        = SL_MONEY;    // Mode basket SL
input double   InpBasketSLMoney    = 30.0;         // Basket SL (uang akun)
input double   InpBasketSLPercent  = 3.0;          // Basket SL (% balance)

//--- Exit: Basket Trailing -----------------------------------------
input group           "=== EXIT: BASKET TRAILING ==="
input bool     InpUseTrailing      = true;         // Trailing pada profit basket
input double   InpTrailStartMoney  = 3.0;          // Mulai trailing saat profit >= ini (uang)
input double   InpTrailStepMoney   = 1.5;          // Jarak trailing dari puncak (uang)

//--- Filter Spread & Waktu -----------------------------------------
input group           "=== FILTER SPREAD & WAKTU ==="
input int      InpMaxSpreadPoints  = 300;          // Spread maksimum (points)
input bool     InpUseTimeFilter    = false;        // Aktifkan filter jam (server time)
input int      InpStartHour        = 0;            // Jam mulai trading (0-23)
input int      InpEndHour          = 24;           // Jam berhenti entry (0-24)
input bool     InpTradeMonday      = true;         // Trading hari Senin
input bool     InpTradeFriday      = true;         // Trading hari Jumat
input bool     InpTradeWeekend     = false;        // Trading Sabtu & Minggu (crypto = true)
input bool     InpFridayForceClose = true;         // Tutup semua sebelum weekend (forex/gold)
input int      InpFridayCloseHour  = 22;           // Jam tutup paksa hari Jumat

//--- Batas Harian ---------------------------------------------------
input group           "=== BATAS HARIAN ==="
input bool     InpUseDailyLimit    = false;        // Aktifkan batas profit/loss harian
input double   InpDailyProfitStop  = 50.0;         // Stop entry jika profit harian >= ini
input double   InpDailyLossStop     = 50.0;        // Stop entry jika loss harian >= ini

//==================================================================//
//  GLOBAL STATE                                                    //
//==================================================================//
CTrade         trade;
CPositionInfo  pos;
CSymbolInfo    sym;

int      hFastMA = INVALID_HANDLE;
int      hSlowMA = INVALID_HANDLE;
int      hRSI    = INVALID_HANDLE;
int      hATR    = INVALID_HANDLE;

datetime gLastEntryTime = 0;
double   gBasketPeakProfit = 0.0;   // puncak profit basket utk trailing
int      gTrailActive = 0;

double   gDayStartBalance = 0.0;
int      gDayStamp = -1;

//==================================================================//
//  INIT / DEINIT                                                   //
//==================================================================//
int OnInit()
{
   if(!sym.Name(_Symbol))
   {
      Print("Gagal init SymbolInfo untuk ", _Symbol);
      return(INIT_FAILED);
   }
   sym.Refresh();

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetAsyncMode(false);

   if(InpUseMAFilter)
   {
      hFastMA = iMA(_Symbol, InpSignalTF, InpFastEMA, 0, MODE_EMA, PRICE_CLOSE);
      hSlowMA = iMA(_Symbol, InpSignalTF, InpSlowEMA, 0, MODE_EMA, PRICE_CLOSE);
      if(hFastMA==INVALID_HANDLE || hSlowMA==INVALID_HANDLE)
      { Print("Gagal buat handle EMA"); return(INIT_FAILED); }
   }
   if(InpUseRSIFilter)
   {
      hRSI = iRSI(_Symbol, InpSignalTF, InpRSIPeriod, PRICE_CLOSE);
      if(hRSI==INVALID_HANDLE) { Print("Gagal buat handle RSI"); return(INIT_FAILED); }
   }
   hATR = iATR(_Symbol, InpATRTF, InpATRPeriod);
   if(hATR==INVALID_HANDLE) { Print("Gagal buat handle ATR"); return(INIT_FAILED); }

   ResetDailyAnchor();

   PrintFormat("BasketScalper siap | %s | magic=%d | target=%s",
               _Symbol, (int)InpMagic,
               (InpTargetMode==TGT_MONEY ?
                  StringFormat("%.2f uang", InpTargetMoney) :
                  StringFormat("%.2f%%", InpTargetPercent)));
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(hFastMA!=INVALID_HANDLE) IndicatorRelease(hFastMA);
   if(hSlowMA!=INVALID_HANDLE) IndicatorRelease(hSlowMA);
   if(hRSI!=INVALID_HANDLE)    IndicatorRelease(hRSI);
   if(hATR!=INVALID_HANDLE)    IndicatorRelease(hATR);
}

//==================================================================//
//  MAIN TICK                                                       //
//==================================================================//
void OnTick()
{
   if(!sym.RefreshRates()) return;

   RollDailyAnchor();

   int    legs   = CountBasket();
   double profit = BasketProfit();

   //--- 1) Kelola exit basket lebih dulu (target / SL / trailing) ---
   if(legs > 0)
   {
      if(ManageBasketExit(profit))
         return;                 // basket ditutup, selesai tick ini
   }
   else
   {
      // Reset state trailing saat tidak ada basket
      gBasketPeakProfit = 0.0;
      gTrailActive = 0;
   }

   //--- 2) Tutup paksa jelang weekend --------------------------------
   if(InpFridayForceClose && IsFridayCloseTime())
   {
      if(legs > 0) CloseBasket("friday-close");
      return;
   }

   //--- 3) Filter sebelum entry --------------------------------------
   if(!IsTradeAllowedNow())        return;
   if(!SpreadOK())                 return;
   if(legs >= InpMaxPositions)     return;
   if(InpUseDailyLimit && DailyLimitHit()) return;

   //--- 4) Sinyal & spacing ------------------------------------------
   int signal = GetSignal();        // +1 BUY, -1 SELL, 0 none
   if(signal == 0) return;

   if(!DirectionAllowed(signal))   return;
   if(!EntryCooldownPassed())      return;
   if(legs > 0 && !SpacingOK(signal)) return;
   if(legs > 0 && !SameDirectionAsBasket(signal)) return; // basket searah

   OpenLeg(signal);
}

//==================================================================//
//  SIGNAL                                                          //
//==================================================================//
// return +1 = BUY, -1 = SELL, 0 = none
int GetSignal()
{
   int maSig = 0;
   double rsiVal = 50.0;

   if(InpUseRSIFilter)
   {
      double rsi[];
      ArraySetAsSeries(rsi, true);
      if(CopyBuffer(hRSI,0,0,1,rsi) < 1) return 0;
      rsiVal = rsi[0];   // index 0 = candle terkini
   }

   if(InpUseMAFilter)
   {
      double fast[], slow[];
      ArraySetAsSeries(fast, true);
      ArraySetAsSeries(slow, true);
      if(CopyBuffer(hFastMA,0,0,2,fast) < 2) return 0;
      if(CopyBuffer(hSlowMA,0,0,2,slow) < 2) return 0;
      if(fast[0] > slow[0]) maSig = 1;
      else if(fast[0] < slow[0]) maSig = -1;
   }

   // Gabungan: MA menentukan arah, RSI menjadi filter izin
   if(InpUseMAFilter && InpUseRSIFilter)
   {
      if(maSig==1  && rsiVal < InpRSIBuyMax)  return 1;
      if(maSig==-1 && rsiVal > InpRSISellMin) return -1;
      return 0;
   }
   if(InpUseMAFilter)  return maSig;
   if(InpUseRSIFilter)
   {
      if(rsiVal < InpRSISellMin) return 1;   // oversold -> buy
      if(rsiVal > InpRSIBuyMax)  return -1;  // overbought -> sell
      return 0;
   }
   return 0;
}

//==================================================================//
//  BASKET HELPERS                                                  //
//==================================================================//
int CountBasket()
{
   int n = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(pos.SelectByIndex(i) &&
         pos.Symbol()==_Symbol && pos.Magic()==InpMagic)
         n++;
   }
   return n;
}

double BasketProfit()
{
   double p = 0.0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(pos.SelectByIndex(i) &&
         pos.Symbol()==_Symbol && pos.Magic()==InpMagic)
         p += pos.Profit() + pos.Swap();
   }
   return p;
}

// Arah basket saat ini: +1 buy, -1 sell, 0 kosong
int BasketDirection()
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(pos.SelectByIndex(i) &&
         pos.Symbol()==_Symbol && pos.Magic()==InpMagic)
         return (pos.PositionType()==POSITION_TYPE_BUY ? 1 : -1);
   }
   return 0;
}

bool SameDirectionAsBasket(int signal)
{
   int d = BasketDirection();
   if(d==0) return true;
   return (d==signal);
}

// Harga entry terakhir (untuk spacing)
double LastEntryPrice()
{
   double price = 0.0;
   datetime latest = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(pos.SelectByIndex(i) &&
         pos.Symbol()==_Symbol && pos.Magic()==InpMagic)
      {
         if(pos.Time() >= latest)
         {
            latest = pos.Time();
            price  = pos.PriceOpen();
         }
      }
   }
   return price;
}

//==================================================================//
//  EXIT MANAGEMENT                                                 //
//==================================================================//
// return true bila basket ditutup
bool ManageBasketExit(double profit)
{
   //--- Target profit ---
   double target = (InpTargetMode==TGT_MONEY)
                   ? InpTargetMoney
                   : AccountInfoDouble(ACCOUNT_BALANCE) * InpTargetPercent / 100.0;
   if(profit >= target && target > 0.0)
   {
      CloseBasket("target-profit");
      return true;
   }

   //--- Basket stop loss ---
   if(InpUseBasketSL)
   {
      double slmoney = (InpSLMode==SL_MONEY)
                       ? InpBasketSLMoney
                       : AccountInfoDouble(ACCOUNT_BALANCE) * InpBasketSLPercent / 100.0;
      if(slmoney > 0.0 && profit <= -slmoney)
      {
         CloseBasket("basket-sl");
         return true;
      }
   }

   //--- Basket trailing ---
   if(InpUseTrailing)
   {
      if(!gTrailActive && profit >= InpTrailStartMoney && InpTrailStartMoney > 0.0)
      {
         gTrailActive = 1;
         gBasketPeakProfit = profit;
      }
      if(gTrailActive)
      {
         if(profit > gBasketPeakProfit) gBasketPeakProfit = profit;
         if(profit <= gBasketPeakProfit - InpTrailStepMoney)
         {
            CloseBasket("basket-trail");
            return true;
         }
      }
   }
   return false;
}

void CloseBasket(string reason)
{
   int closed = 0;
   // Loop berulang: penutupan bisa gagal parsial, ulangi sampai bersih
   for(int attempt = 0; attempt < 5; attempt++)
   {
      bool anyLeft = false;
      for(int i = PositionsTotal()-1; i >= 0; i--)
      {
         if(!pos.SelectByIndex(i)) continue;
         if(pos.Symbol()!=_Symbol || pos.Magic()!=InpMagic) continue;
         anyLeft = true;
         if(trade.PositionClose(pos.Ticket()))
            closed++;
      }
      if(!anyLeft) break;
   }
   gBasketPeakProfit = 0.0;
   gTrailActive = 0;
   gLastEntryTime = 0;
   PrintFormat("Basket ditutup (%s) — %d posisi", reason, closed);
}

//==================================================================//
//  ENTRY                                                           //
//==================================================================//
void OpenLeg(int signal)
{
   double lot = ComputeLot();
   if(lot <= 0.0) return;

   double price, sl = 0.0, tp = 0.0;   // SL/TP per-order 0: exit dikelola basket
   bool ok = false;
   if(signal > 0)
   {
      price = sym.Ask();
      ok = trade.Buy(lot, _Symbol, price, sl, tp, InpComment);
   }
   else
   {
      price = sym.Bid();
      ok = trade.Sell(lot, _Symbol, price, sl, tp, InpComment);
   }

   if(ok)
   {
      gLastEntryTime = TimeCurrent();
      PrintFormat("Entry %s lot=%.2f @%.5f (ret=%d)",
                  (signal>0?"BUY":"SELL"), lot, price, trade.ResultRetcode());
   }
   else
   {
      PrintFormat("Entry GAGAL %s: ret=%d %s",
                  (signal>0?"BUY":"SELL"), trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
   }
}

double ComputeLot()
{
   double lot;
   if(InpLotMode==LOT_FIXED)
   {
      lot = InpFixedLot;
   }
   else
   {
      // Risk % balance dgn jarak SL InpRiskSLPoints
      double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
      double riskMoney = balance * InpRiskPercent / 100.0;
      double tickVal   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double point     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      if(tickVal<=0 || tickSize<=0 || point<=0) return NormalizeLot(InpFixedLot);
      double slPriceDist = InpRiskSLPoints * point;
      double valuePerLot = (slPriceDist / tickSize) * tickVal; // rugi per 1.0 lot di SL
      if(valuePerLot <= 0) return NormalizeLot(InpFixedLot);
      lot = riskMoney / valuePerLot;
   }
   return NormalizeLot(lot);
}

double NormalizeLot(double lot)
{
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(stepLot <= 0) stepLot = 0.01;
   lot = MathFloor(lot/stepLot) * stepLot;
   if(lot < minLot) lot = minLot;
   if(lot > maxLot) lot = maxLot;
   return NormalizeDouble(lot, 2);
}

//==================================================================//
//  FILTERS                                                         //
//==================================================================//
bool SpreadOK()
{
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(InpMaxSpreadPoints > 0 && spread > InpMaxSpreadPoints)
      return false;
   return true;
}

bool DirectionAllowed(int signal)
{
   if(InpDirectionMode==DIR_BUY_ONLY)  return (signal>0);
   if(InpDirectionMode==DIR_SELL_ONLY) return (signal<0);
   return true; // DIR_SIGNAL
}

bool EntryCooldownPassed()
{
   if(InpEntryCooldownSec <= 0) return true;
   return (TimeCurrent() - gLastEntryTime >= InpEntryCooldownSec);
}

bool SpacingOK(int signal)
{
   if(!InpUseSpacing) return true;
   double last = LastEntryPrice();
   if(last <= 0.0) return true;

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double stepPrice;
   if(InpStepMode==STEP_ATR)
   {
      double atr[];
      ArraySetAsSeries(atr, true);
      if(CopyBuffer(hATR,0,0,1,atr) < 1) return true;
      stepPrice = atr[0] * InpStepATRmult;
   }
   else
   {
      stepPrice = InpStepPoints * point;
   }
   double cur = (signal>0 ? sym.Ask() : sym.Bid());
   return (MathAbs(cur - last) >= stepPrice);
}

bool IsTradeAllowedNow()
{
   if(!InpUseTimeFilter) return WeekdayAllowed();
   if(!WeekdayAllowed()) return false;

   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   int h = t.hour;
   if(InpStartHour <= InpEndHour)
      return (h >= InpStartHour && h < InpEndHour);
   // rentang melewati tengah malam
   return (h >= InpStartHour || h < InpEndHour);
}

bool WeekdayAllowed()
{
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   if(t.day_of_week==1 && !InpTradeMonday) return false;
   if(t.day_of_week==5 && !InpTradeFriday) return false;
   // Sabtu (6) & Minggu (0): hanya jika InpTradeWeekend aktif (mis. crypto)
   if((t.day_of_week==0 || t.day_of_week==6) && !InpTradeWeekend) return false;
   return true;
}

bool IsFridayCloseTime()
{
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   return (t.day_of_week==5 && t.hour >= InpFridayCloseHour);
}

//==================================================================//
//  DAILY LIMIT                                                     //
//==================================================================//
void ResetDailyAnchor()
{
   gDayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   gDayStamp = t.day_of_year;
}

void RollDailyAnchor()
{
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   if(t.day_of_year != gDayStamp)
      ResetDailyAnchor();
}

bool DailyLimitHit()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double pnl    = equity - gDayStartBalance;   // realized+floating hari ini (perkiraan)
   if(InpDailyProfitStop > 0.0 && pnl >=  InpDailyProfitStop) return true;
   if(InpDailyLossStop   > 0.0 && pnl <= -InpDailyLossStop)   return true;
   return false;
}
//+------------------------------------------------------------------+
