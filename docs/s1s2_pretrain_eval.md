# S1+S2 on the 34 seen target tasks in pretrain kitchens (development tasks)

Source: sys1_sys2 `doc/results.md` (origin/master, commit eb3b0da), "Pretrain-scene reference run": the official
RoboCasa365 client protocol with `split="pretrain"`, 50 sequential trials per task. S1 is pi0.5 progact step 269999 and S2 is
Qwen3-VL-4B checkpoint-17124, with language steering on. Overall 54.3 (atomic_seen 80.1, composite_seen 51.5, composite_unseen 28.1);
the same models in the target scenes reach 51.4 (78.8 / 49.4 / 22.6). The Harness+S1 column is `harness_check_full2` (Ray, Astra +
S1 with task notes, seeds 0-2 in HexaShell's pretrain50 kitchens). There is no S1+S2 run on the 93 non-target development tasks.

| task | set | horizon | S1+S2 pretrain scenes (%) | S1+S2 target scenes (%) | Harness+S1 |
|---|---|---|---|---|---|
| PrepareCoffee | composite_seen | 1800 | 18 | 12 | 3/3 |
| PackIdenticalLunches | composite_seen | 3900 | 24 | 20 | 3/3 |
| SearingMeat | composite_seen | 4350 | 28 | 36 | 3/3 |
| StirVegetables | composite_seen | 2400 | 36 | 42 | 3/3 |
| CloseBlenderLid | atomic_seen | 900 | 42 | 48 | 2/3 |
| RinseSinkBasin | composite_seen | 1350 | 42 | 50 | 2/3 |
| KettleBoiling | composite_seen | 1500 | 46 | 54 | 3/3 |
| CoffeeSetupMug | atomic_seen | 600 | 50 | 54 | 2/3 |
| SteamInMicrowave | composite_seen | 2100 | 50 | 32 | 2/3 |
| GetToastedBread | composite_seen | 3000 | 56 | 74 | 3/3 |
| DeliverStraw | composite_seen | 2550 | 58 | 56 | 3/3 |
| TurnOffStove | atomic_seen | 750 | 58 | 44 | 3/3 |
| StoreLeftoversInBowl | composite_seen | 2550 | 60 | 40 | 3/3 |
| PreSoakPan | composite_seen | 2400 | 62 | 70 | 2/3 |
| WashLettuce | composite_seen | 1650 | 62 | 62 | 3/3 |
| SetUpCuttingStation | composite_seen | 2400 | 64 | 58 | 3/3 |
| LoadDishwasher | composite_seen | 1800 | 70 | 46 | 3/3 |
| StackBowlsCabinet | composite_seen | 2100 | 72 | 76 | 3/3 |
| TurnOnSinkFaucet | atomic_seen | 600 | 72 | 66 | 3/3 |
| NavigateKitchen | atomic_seen | 450 | 74 | 60 | 2/3 |
| PickPlaceCounterToCabinet | atomic_seen | 750 | 74 | 76 | 3/3 |
| ScrubCuttingBoard | composite_seen | 1200 | 76 | 62 | 3/3 |
| OpenDrawer | atomic_seen | 750 | 80 | 94 | 3/3 |
| PickPlaceCounterToStove | atomic_seen | 600 | 86 | 88 | 3/3 |
| SlideDishwasherRack | atomic_seen | 450 | 86 | 78 | 3/3 |
| TurnOnMicrowave | atomic_seen | 450 | 86 | 82 | 3/3 |
| PickPlaceDrawerToCounter | atomic_seen | 750 | 88 | 74 | 3/3 |
| OpenCabinet | atomic_seen | 1050 | 90 | 96 | 3/3 |
| PickPlaceToasterToCounter | atomic_seen | 600 | 90 | 90 | 3/3 |
| CloseToasterOvenDoor | atomic_seen | 450 | 92 | 84 | 3/3 |
| PickPlaceSinkToCounter | atomic_seen | 900 | 92 | 100 | 3/3 |
| TurnOnElectricKettle | atomic_seen | 450 | 92 | 92 | 3/3 |
| OpenStandMixerHead | atomic_seen | 450 | 94 | 100 | 3/3 |
| CloseFridge | atomic_seen | 900 | 96 | 92 | 3/3 |
