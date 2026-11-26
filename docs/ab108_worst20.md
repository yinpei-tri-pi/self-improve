# The 20 weakest of the 108 tasks, with the current notes

Ranked by success with the current notes: the A/B's `with` arm (5 seeds), or the re-check on 6 fresh seeds for the
five tasks (*) whose notes were revised after the A/B; ties are broken by the `without` arm. Failures are counted under the current notes:
*drop* = an object slipped out of the gripper and was lost; *steps* = the step budget ran out. `scripts/ab_compare.py`
has the full per-task tables (`docs/ab108.md`, `docs/ab108r2.md`).

| # | task | kind / family | horizon | current notes | A/B with | A/B without | re-check without | pilot | failures | median steps (successes) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FreezeIceTray | composite / managing_freezer_space | 2400 | **2/5** | 2/5 | 2/5 | - | 0/2 | steps 2, drop 1 | 1645 |
| 2 | CloseOven * | atomic | 450 | **3/6** | 3/5 | 5/5 | 3/6 | 2/2 | steps 3 | 235 |
| 3 | SetupWineGlasses | composite / setting_the_table | 3150 | **3/5** | 3/5 | 1/5 | - | n/a | drop 1, steps 1 | 2667 |
| 4 | AlignSilverware | composite / setting_the_table | 2550 | **3/5** | 3/5 | 2/5 | - | 2/2 | steps 1, drop 1 | 1848 |
| 5 | AddLemonToFish | composite / garnishing_dishes | 2100 | **3/5** | 3/5 | 3/5 | - | 1/2 | drop 2 | 1848 |
| 6 | CoffeeServeMug | atomic | 450 | **3/5** | 3/5 | 3/5 | - | 0/2 | steps 2 | 255 |
| 7 | PickPlaceCounterToOven | atomic | 600 | **3/5** | 3/5 | 3/5 | - | 2/2 | steps 2 | 353 |
| 8 | TurnOffMicrowave | atomic | 300 | **3/5** | 3/5 | 3/5 | - | 2/2 | steps 2 | 159 |
| 9 | DeliverBrewedCoffee | composite / brewing | 1050 | **3/5** | 3/5 | 4/5 | - | 1/2 | drop 1, steps 1 | 571 |
| 10 | PreheatOven | atomic | 450 | **3/5** | 3/5 | 4/5 | - | 2/2 | steps 2 | 193 |
| 11 | ScalePortioning * | composite / portioning_meals | 2100 | **4/6** | 0/5 | 2/5 | 5/6 | 2/2 | drop 2 | 1860 |
| 12 | PrewashFoodAssembly | composite / washing_fruits_and_vegetables | 1650 | **4/5** | 4/5 | 2/5 | - | 2/2 | steps 1 | 1153 |
| 13 | DistributeChicken | composite / portioning_meals | 2400 | **4/5** | 4/5 | 3/5 | - | 1/2 | drop 1 | 1646 |
| 14 | DrainVeggies | composite / washing_fruits_and_vegetables | 2400 | **4/5** | 4/5 | 3/5 | - | 2/2 | drop 1 | 1854 |
| 15 | FreezeBottledWaters | composite / managing_freezer_space | 3900 | **4/5** | 4/5 | 3/5 | - | 1/2 | drop 1 | 1632 |
| 16 | HotDogSetup | composite / preparing_sandwiches | 4200 | **4/5** | 4/5 | 3/5 | - | n/a | drop 1 | 2841 |
| 17 | MaximizeFreezerSpace | composite / managing_freezer_space | 1650 | **4/5** | 4/5 | 3/5 | - | 1/2 | steps 1 | 1063 |
| 18 | AdjustToasterOvenTemperature | atomic | 450 | **4/5** | 4/5 | 4/5 | - | 2/2 | steps 1 | 210 |
| 19 | CheesyBread | atomic | 750 | **4/5** | 4/5 | 4/5 | - | 2/2 | steps 1 | 342 |
| 20 | ChooseMeasuringCup | composite / measuring_ingredients | 1800 | **4/5** | 4/5 | 4/5 | - | 2/2 | steps 1 | 1734 |
