# All 108 selected (non-target) tasks, worst to best

Success with the current task notes: the A/B's `with` arm (5 seeds), or for the 5 tasks whose notes were revised (*)
the re-check on 6 fresh seeds. Ties are broken by the `without` arm, then by name. Rows above the line are the 20 dropped from the
rollouts (`assets/rollout_tasks.json`). Failures under the current notes: drop = an object slipped from the gripper and was lost;
steps = the step budget ran out.

| # | task | kind / family | horizon | current notes | A/B with | A/B without | pilot | failures |
|---|---|---|---|---|---|---|---|---|
| 1 | FreezeIceTray | composite / managing_freezer_space | 2400 | **2/5** | 2/5 | 2/5 | 0/2 | steps 2, drop 1 |
| 2 | CloseOven * | atomic | 450 | **3/6** | 3/5 | 5/5 | 2/2 | steps 3 |
| 3 | SetupWineGlasses | composite / setting_the_table | 3150 | **3/5** | 3/5 | 1/5 | n/a | drop 1, steps 1 |
| 4 | AlignSilverware | composite / setting_the_table | 2550 | **3/5** | 3/5 | 2/5 | 2/2 | steps 1, drop 1 |
| 5 | AddLemonToFish | composite / garnishing_dishes | 2100 | **3/5** | 3/5 | 3/5 | 1/2 | drop 2 |
| 6 | CoffeeServeMug | atomic | 450 | **3/5** | 3/5 | 3/5 | 0/2 | steps 2 |
| 7 | PickPlaceCounterToOven | atomic | 600 | **3/5** | 3/5 | 3/5 | 2/2 | steps 2 |
| 8 | TurnOffMicrowave | atomic | 300 | **3/5** | 3/5 | 3/5 | 2/2 | steps 2 |
| 9 | DeliverBrewedCoffee | composite / brewing | 1050 | **3/5** | 3/5 | 4/5 | 1/2 | drop 1, steps 1 |
| 10 | PreheatOven | atomic | 450 | **3/5** | 3/5 | 4/5 | 2/2 | steps 2 |
| 11 | ScalePortioning * | composite / portioning_meals | 2100 | **4/6** | 0/5 | 2/5 | 2/2 | drop 2 |
| 12 | PrewashFoodAssembly | composite / washing_fruits_and_vegetables | 1650 | **4/5** | 4/5 | 2/5 | 2/2 | steps 1 |
| 13 | DistributeChicken | composite / portioning_meals | 2400 | **4/5** | 4/5 | 3/5 | 1/2 | drop 1 |
| 14 | DrainVeggies | composite / washing_fruits_and_vegetables | 2400 | **4/5** | 4/5 | 3/5 | 2/2 | drop 1 |
| 15 | FreezeBottledWaters | composite / managing_freezer_space | 3900 | **4/5** | 4/5 | 3/5 | 1/2 | drop 1 |
| 16 | HotDogSetup | composite / preparing_sandwiches | 4200 | **4/5** | 4/5 | 3/5 | n/a | drop 1 |
| 17 | MaximizeFreezerSpace | composite / managing_freezer_space | 1650 | **4/5** | 4/5 | 3/5 | 1/2 | steps 1 |
| 18 | AdjustToasterOvenTemperature | atomic | 450 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 19 | CheesyBread | atomic | 750 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 20 | ChooseMeasuringCup | composite / measuring_ingredients | 1800 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| | *(dropped above)* | | | | | | | |
| 21 | CloseFridgeDrawer | atomic | 450 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 22 | GarnishCupcake | composite / garnishing_dishes | 1800 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 23 | HeatMug | composite / reheating_food | 1800 | **4/5** | 4/5 | 4/5 | 1/2 | drop 1 |
| 24 | MakeLoadedPotato | composite / reheating_food | 3000 | **4/5** | 4/5 | 4/5 | 2/2 | drop 1 |
| 25 | OrganizeCondiments | composite / arranging_condiments | 2250 | **4/5** | 4/5 | 4/5 | 2/2 | drop 1 |
| 26 | PickPlaceCabinetToCounter | atomic | 450 | **4/5** | 4/5 | 4/5 | 2/2 | drop 1 |
| 27 | PickPlaceCounterToMicrowave | atomic | 1050 | **4/5** | 4/5 | 4/5 | 2/2 | drop 1 |
| 28 | PickPlaceCounterToToasterOven | atomic | 450 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 29 | PickPlaceFridgeDrawerToShelf | atomic | 600 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 30 | PlaceFoodInBowls | composite / serving_food | 2550 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 31 | PortionOnSize | composite / portioning_meals | 2700 | **4/5** | 4/5 | 4/5 | 1/2 | steps 1 |
| 32 | PrewashFoodSorting | composite / washing_fruits_and_vegetables | 3600 | **4/5** | 4/5 | 4/5 | 1/2 | drop 1 |
| 33 | RecycleSodaCans | composite / organizing_recycling | 6750 | **4/5** | 4/5 | 4/5 | 2/2 | drop 1 |
| 34 | StartCoffeeMachine | atomic | 300 | **4/5** | 4/5 | 4/5 | 2/2 | steps 1 |
| 35 | CloseElectricKettleLid | atomic | 300 | **4/5** | 4/5 | 5/5 | 2/2 | steps 1 |
| 36 | OpenBlenderLid | atomic | 450 | **4/5** | 4/5 | 5/5 | 2/2 | drop 1 |
| 37 | OpenElectricKettleLid | atomic | 300 | **4/5** | 4/5 | 5/5 | 2/2 | steps 1 |
| 38 | PickPlaceStoveToCounter | atomic | 450 | **4/5** | 4/5 | 5/5 | 2/2 | drop 1 |
| 39 | RetrieveIceTray | composite / adding_ice_to_beverages | 2550 | **4/5** | 4/5 | 5/5 | 1/2 | drop 1 |
| 40 | SetupFruitBowl | composite / setting_the_table | 3750 | **4/5** | 4/5 | 5/5 | 2/2 | drop 1 |
| 41 | MoveToFreezerDrawer * | composite / managing_freezer_space | 1800 | **5/6** | 2/5 | 4/5 | 2/2 | drop 1 |
| 42 | SlideOvenRack * | atomic | 450 | **5/6** | 3/5 | 5/5 | 2/2 | drop 1 |
| 43 | RecycleBottlesBySize | composite / organizing_recycling | 3750 | **5/5** | 5/5 | 2/5 | 2/2 | - |
| 44 | ReorganizeFrozenVegetables | composite / managing_freezer_space | 2700 | **5/5** | 5/5 | 2/5 | 0/2 | - |
| 45 | ArrangeBreadBowl | composite / setting_the_table | 1350 | **5/5** | 5/5 | 3/5 | 1/2 | - |
| 46 | ClearFreezer | composite / managing_freezer_space | 3600 | **5/5** | 5/5 | 3/5 | 2/2 | - |
| 47 | CloseDishwasher | atomic | 300 | **5/5** | 5/5 | 3/5 | 2/2 | - |
| 48 | DateNight | composite / setting_the_table | 4800 | **5/5** | 5/5 | 3/5 | 2/2 | - |
| 49 | LineUpCondiments | composite / arranging_condiments | 3000 | **5/5** | 5/5 | 3/5 | 1/2 | - |
| 50 | SweetenCoffee | composite / brewing | 1950 | **5/5** | 5/5 | 3/5 | 1/2 | - |
| 51 | AdjustWaterTemperature | atomic | 450 | **5/5** | 5/5 | 4/5 | 1/2 | - |
| 52 | BeverageOrganization * | composite / setting_the_table | 4500 | **6/6** | 1/5 | 4/5 | 0/2 | - |
| 53 | CloseCabinet | atomic | 750 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 54 | CloseMicrowave | atomic | 450 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 55 | DessertUpgrade | composite / serving_food | 1200 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 56 | OpenDishwasher | atomic | 450 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 57 | OpenOven | atomic | 300 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 58 | OpenToasterOvenDoor | atomic | 450 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 59 | PickPlaceToasterOvenToCounter | atomic | 450 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 60 | PlaceEqualIceCubes | composite / adding_ice_to_beverages | 2850 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 61 | PrepareToast | composite / making_toast | 3750 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 62 | SetupBowls | composite / setting_the_table | 2700 | **5/5** | 5/5 | 4/5 | 1/2 | - |
| 63 | SimmeringSauce | composite / reheating_food | 3450 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 64 | SlideToasterOvenRack | atomic | 300 | **5/5** | 5/5 | 4/5 | 1/2 | - |
| 65 | SweetSavoryToastSetup | composite / making_toast | 4050 | **5/5** | 5/5 | 4/5 | 2/2 | - |
| 66 | AddSugarCubes | composite / garnishing_dishes | 900 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 67 | AfterwashSorting | composite / washing_fruits_and_vegetables | 3000 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 68 | AirDryFruit | composite / washing_fruits_and_vegetables | 2850 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 69 | AlcoholServingPrep | composite / serving_food | 4650 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 70 | ArrangeDrinkware | composite / setting_the_table | 1650 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 71 | ClearClutter | composite / washing_fruits_and_vegetables | 3300 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 72 | ClearSinkSpace | composite / washing_fruits_and_vegetables | 1800 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 73 | CloseDrawer | atomic | 450 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 74 | CloseStandMixerHead | atomic | 450 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 75 | GatherCuttingTools | composite / chopping_vegetables | 2400 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 76 | GatherVegetables | composite / preparing_sandwiches | 1350 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 77 | LowerHeat | atomic | 750 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 78 | MakeIcedCoffee | atomic | 750 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 79 | MoveFridgeToFreezer | composite / managing_freezer_space | 1650 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 80 | OpenFridge | atomic | 900 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 81 | OpenFridgeDrawer | atomic | 450 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 82 | OpenMicrowave | atomic | 600 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 83 | OrganizeCoffeeCondiments | composite / brewing | 1950 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 84 | PackDessert | atomic | 600 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 85 | PickPlaceCounterToDrawer | atomic | 750 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 86 | PickPlaceCounterToSink | atomic | 600 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 87 | PickPlaceFridgeShelfToDrawer | atomic | 600 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 88 | PickPlaceMicrowaveToCounter | atomic | 750 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 89 | PlaceIceInCup | composite / adding_ice_to_beverages | 1200 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 90 | PortionFruitBowl | composite / portioning_meals | 2100 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 91 | PortionInTupperware | composite / portioning_meals | 1950 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 92 | PortionYogurt | composite / portioning_meals | 2100 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 93 | PrepareSausageCheese | composite / preparing_sandwiches | 3600 | **5/5** | 5/5 | 5/5 | n/a | - |
| 94 | PrepareSoupServing | composite / serving_food | 3450 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 95 | RecycleStackedYogurt | composite / organizing_recycling | 2100 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 96 | ResetCabinetDoors | composite / arranging_cabinets | 3300 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 97 | SeasoningSpiceSetup | composite / setting_the_table | 4200 | **5/5** | 5/5 | 5/5 | 1/2 | - |
| 98 | ServeSteak | composite / serving_food | 2100 | **5/5** | 5/5 | 5/5 | 1/2 | - |
| 99 | SetBowlsForSoup | composite / setting_the_table | 5400 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 100 | SetupButterPlate | composite / setting_the_table | 4050 | **5/5** | 5/5 | 5/5 | n/a | - |
| 101 | SizeSorting | composite / setting_the_table | 1500 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 102 | StackCans | composite / arranging_cabinets | 1800 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 103 | TurnOffSinkFaucet | atomic | 300 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 104 | TurnOnStove | atomic | 450 | **5/5** | 5/5 | 5/5 | 1/2 | - |
| 105 | TurnOnToaster | atomic | 300 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 106 | TurnOnToasterOven | atomic | 450 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 107 | TurnSinkSpout | atomic | 300 | **5/5** | 5/5 | 5/5 | 2/2 | - |
| 108 | WarmCroissant | composite / reheating_food | 1650 | **5/5** | 5/5 | 5/5 | 1/2 | - |
