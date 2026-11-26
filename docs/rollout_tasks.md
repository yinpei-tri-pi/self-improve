# Rollout tasks (120)

The 34 target tasks of atomic_seen, composite_seen, run in RoboCasa's pretrain kitchens with the
pretrain objects (never the target split's), plus 86 of the 108 selected tasks (`docs/task_selection.md`): every selected task except
the 22 weakest composite ones in the prompt A/B (`docs/ab108_findings.md`). Written by `scripts/rollout_tasks.py`;`rollout.tasks: rollout` plays them.

| task set | tasks |
|---|---|
| atomic_seen | 18 |
| composite_seen | 16 |
| pretrain_atomic | 44 |
| pretrain_composite | 42 |

## Dropped (22)

| # | task | current notes | A/B with | A/B without |
|---|---|---|---|---|
| 1 | FreezeIceTray | 2/5 | 2/5 | 2/5 |
| 2 | SetupWineGlasses | 3/5 | 3/5 | 1/5 |
| 3 | AlignSilverware | 3/5 | 3/5 | 2/5 |
| 4 | AddLemonToFish | 3/5 | 3/5 | 3/5 |
| 5 | DeliverBrewedCoffee | 3/5 | 3/5 | 4/5 |
| 6 | ScalePortioning (revised notes) | 4/6 | 0/5 | 2/5 |
| 7 | PrewashFoodAssembly | 4/5 | 4/5 | 2/5 |
| 8 | DistributeChicken | 4/5 | 4/5 | 3/5 |
| 9 | DrainVeggies | 4/5 | 4/5 | 3/5 |
| 10 | FreezeBottledWaters | 4/5 | 4/5 | 3/5 |
| 11 | HotDogSetup | 4/5 | 4/5 | 3/5 |
| 12 | MaximizeFreezerSpace | 4/5 | 4/5 | 3/5 |
| 13 | ChooseMeasuringCup | 4/5 | 4/5 | 4/5 |
| 14 | GarnishCupcake | 4/5 | 4/5 | 4/5 |
| 15 | HeatMug | 4/5 | 4/5 | 4/5 |
| 16 | MakeLoadedPotato | 4/5 | 4/5 | 4/5 |
| 17 | OrganizeCondiments | 4/5 | 4/5 | 4/5 |
| 18 | PlaceFoodInBowls | 4/5 | 4/5 | 4/5 |
| 19 | PortionOnSize | 4/5 | 4/5 | 4/5 |
| 20 | PrewashFoodSorting | 4/5 | 4/5 | 4/5 |
| 21 | RecycleSodaCans | 4/5 | 4/5 | 4/5 |
| 22 | RetrieveIceTray | 4/5 | 4/5 | 5/5 |

## All rollout tasks

| task | source | task set | horizon |
|---|---|---|---|
| CloseBlenderLid | target | atomic_seen | 900 |
| CloseFridge | target | atomic_seen | 900 |
| CloseToasterOvenDoor | target | atomic_seen | 450 |
| CoffeeSetupMug | target | atomic_seen | 600 |
| DeliverStraw | target | composite_seen | 2550 |
| GetToastedBread | target | composite_seen | 3000 |
| KettleBoiling | target | composite_seen | 1500 |
| LoadDishwasher | target | composite_seen | 1800 |
| NavigateKitchen | target | atomic_seen | 450 |
| OpenCabinet | target | atomic_seen | 1050 |
| OpenDrawer | target | atomic_seen | 750 |
| OpenStandMixerHead | target | atomic_seen | 450 |
| PackIdenticalLunches | target | composite_seen | 3900 |
| PickPlaceCounterToCabinet | target | atomic_seen | 750 |
| PickPlaceCounterToStove | target | atomic_seen | 600 |
| PickPlaceDrawerToCounter | target | atomic_seen | 750 |
| PickPlaceSinkToCounter | target | atomic_seen | 900 |
| PickPlaceToasterToCounter | target | atomic_seen | 600 |
| PreSoakPan | target | composite_seen | 2400 |
| PrepareCoffee | target | composite_seen | 1800 |
| RinseSinkBasin | target | composite_seen | 1350 |
| ScrubCuttingBoard | target | composite_seen | 1200 |
| SearingMeat | target | composite_seen | 4350 |
| SetUpCuttingStation | target | composite_seen | 2400 |
| SlideDishwasherRack | target | atomic_seen | 450 |
| StackBowlsCabinet | target | composite_seen | 2100 |
| SteamInMicrowave | target | composite_seen | 2100 |
| StirVegetables | target | composite_seen | 2400 |
| StoreLeftoversInBowl | target | composite_seen | 2550 |
| TurnOffStove | target | atomic_seen | 750 |
| TurnOnElectricKettle | target | atomic_seen | 450 |
| TurnOnMicrowave | target | atomic_seen | 450 |
| TurnOnSinkFaucet | target | atomic_seen | 600 |
| WashLettuce | target | composite_seen | 1650 |
| AddSugarCubes | selected | pretrain_composite | 900 |
| AdjustToasterOvenTemperature | selected | pretrain_atomic | 450 |
| AdjustWaterTemperature | selected | pretrain_atomic | 450 |
| AfterwashSorting | selected | pretrain_composite | 3000 |
| AirDryFruit | selected | pretrain_composite | 2850 |
| AlcoholServingPrep | selected | pretrain_composite | 4650 |
| ArrangeBreadBowl | selected | pretrain_composite | 1350 |
| ArrangeDrinkware | selected | pretrain_composite | 1650 |
| BeverageOrganization | selected | pretrain_composite | 4500 |
| CheesyBread | selected | pretrain_atomic | 750 |
| ClearClutter | selected | pretrain_composite | 3300 |
| ClearFreezer | selected | pretrain_composite | 3600 |
| ClearSinkSpace | selected | pretrain_composite | 1800 |
| CloseCabinet | selected | pretrain_atomic | 750 |
| CloseDishwasher | selected | pretrain_atomic | 300 |
| CloseDrawer | selected | pretrain_atomic | 450 |
| CloseElectricKettleLid | selected | pretrain_atomic | 300 |
| CloseFridgeDrawer | selected | pretrain_atomic | 450 |
| CloseMicrowave | selected | pretrain_atomic | 450 |
| CloseOven | selected | pretrain_atomic | 450 |
| CloseStandMixerHead | selected | pretrain_atomic | 450 |
| CoffeeServeMug | selected | pretrain_atomic | 450 |
| DateNight | selected | pretrain_composite | 4800 |
| DessertUpgrade | selected | pretrain_composite | 1200 |
| GatherCuttingTools | selected | pretrain_composite | 2400 |
| GatherVegetables | selected | pretrain_composite | 1350 |
| LineUpCondiments | selected | pretrain_composite | 3000 |
| LowerHeat | selected | pretrain_atomic | 750 |
| MakeIcedCoffee | selected | pretrain_atomic | 750 |
| MoveFridgeToFreezer | selected | pretrain_composite | 1650 |
| MoveToFreezerDrawer | selected | pretrain_composite | 1800 |
| OpenBlenderLid | selected | pretrain_atomic | 450 |
| OpenDishwasher | selected | pretrain_atomic | 450 |
| OpenElectricKettleLid | selected | pretrain_atomic | 300 |
| OpenFridge | selected | pretrain_atomic | 900 |
| OpenFridgeDrawer | selected | pretrain_atomic | 450 |
| OpenMicrowave | selected | pretrain_atomic | 600 |
| OpenOven | selected | pretrain_atomic | 300 |
| OpenToasterOvenDoor | selected | pretrain_atomic | 450 |
| OrganizeCoffeeCondiments | selected | pretrain_composite | 1950 |
| PackDessert | selected | pretrain_atomic | 600 |
| PickPlaceCabinetToCounter | selected | pretrain_atomic | 450 |
| PickPlaceCounterToDrawer | selected | pretrain_atomic | 750 |
| PickPlaceCounterToMicrowave | selected | pretrain_atomic | 1050 |
| PickPlaceCounterToOven | selected | pretrain_atomic | 600 |
| PickPlaceCounterToSink | selected | pretrain_atomic | 600 |
| PickPlaceCounterToToasterOven | selected | pretrain_atomic | 450 |
| PickPlaceFridgeDrawerToShelf | selected | pretrain_atomic | 600 |
| PickPlaceFridgeShelfToDrawer | selected | pretrain_atomic | 600 |
| PickPlaceMicrowaveToCounter | selected | pretrain_atomic | 750 |
| PickPlaceStoveToCounter | selected | pretrain_atomic | 450 |
| PickPlaceToasterOvenToCounter | selected | pretrain_atomic | 450 |
| PlaceEqualIceCubes | selected | pretrain_composite | 2850 |
| PlaceIceInCup | selected | pretrain_composite | 1200 |
| PortionFruitBowl | selected | pretrain_composite | 2100 |
| PortionInTupperware | selected | pretrain_composite | 1950 |
| PortionYogurt | selected | pretrain_composite | 2100 |
| PreheatOven | selected | pretrain_atomic | 450 |
| PrepareSausageCheese | selected | pretrain_composite | 3600 |
| PrepareSoupServing | selected | pretrain_composite | 3450 |
| PrepareToast | selected | pretrain_composite | 3750 |
| RecycleBottlesBySize | selected | pretrain_composite | 3750 |
| RecycleStackedYogurt | selected | pretrain_composite | 2100 |
| ReorganizeFrozenVegetables | selected | pretrain_composite | 2700 |
| ResetCabinetDoors | selected | pretrain_composite | 3300 |
| SeasoningSpiceSetup | selected | pretrain_composite | 4200 |
| ServeSteak | selected | pretrain_composite | 2100 |
| SetBowlsForSoup | selected | pretrain_composite | 5400 |
| SetupBowls | selected | pretrain_composite | 2700 |
| SetupButterPlate | selected | pretrain_composite | 4050 |
| SetupFruitBowl | selected | pretrain_composite | 3750 |
| SimmeringSauce | selected | pretrain_composite | 3450 |
| SizeSorting | selected | pretrain_composite | 1500 |
| SlideOvenRack | selected | pretrain_atomic | 450 |
| SlideToasterOvenRack | selected | pretrain_atomic | 300 |
| StackCans | selected | pretrain_composite | 1800 |
| StartCoffeeMachine | selected | pretrain_atomic | 300 |
| SweetSavoryToastSetup | selected | pretrain_composite | 4050 |
| SweetenCoffee | selected | pretrain_composite | 1950 |
| TurnOffMicrowave | selected | pretrain_atomic | 300 |
| TurnOffSinkFaucet | selected | pretrain_atomic | 300 |
| TurnOnStove | selected | pretrain_atomic | 450 |
| TurnOnToaster | selected | pretrain_atomic | 300 |
| TurnOnToasterOven | selected | pretrain_atomic | 450 |
| TurnSinkSpout | selected | pretrain_atomic | 300 |
| WarmCroissant | selected | pretrain_composite | 1650 |
